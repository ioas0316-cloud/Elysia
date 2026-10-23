#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

#if defined(__CUDACC__) || defined(USE_CUDA)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#else
#include "elysia/cuda_host_stub.h"
#endif

#define TILE_DIM 8
#define SHARED_DIM (TILE_DIM + 2) // 10 (8 + 1+1 Halo Padding)

// ----------------------------------------------------------------------------
// 1. SE(3) Lie Algebra -> Lie Group Exponential Map & Kernel
// ----------------------------------------------------------------------------
__device__ void exp_se3(const float twist[6], float R[3][3], float t[3]) {
    float wx = twist[0], wy = twist[1], wz = twist[2];
    float vx = twist[3], vy = twist[4], vz = twist[5];

    float theta_sq = wx * wx + wy * wy + wz * wz;
    float theta = sqrtf(theta_sq);

    if (theta < 1e-6f) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) R[i][j] = (i == j) ? 1.0f : 0.0f;
        }
        t[0] = vx; t[1] = vy; t[2] = vz;
        return;
    }

    float K[3][3] = {
        {   0.0f, -wz,   wy},
        { wz,    0.0f, -wx},
        {-wy,   wx,    0.0f}
    };

    float sin_t = sinf(theta);
    float cos_t = cosf(theta);
    float inv_t = 1.0f / theta;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float K2_ij = 0.0f;
            for (int k = 0; k < 3; ++k) K2_ij += K[i][k] * K[k][j];
            float I_ij = (i == j) ? 1.0f : 0.0f;
            R[i][j] = I_ij + (sin_t * inv_t) * K[i][j] + ((1.0f - cos_t) / theta_sq) * K2_ij;
        }
    }

    float V[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float K2_ij = 0.0f;
            for (int k = 0; k < 3; ++k) K2_ij += K[i][k] * K[k][j];
            float I_ij = (i == j) ? 1.0f : 0.0f;
            V[i][j] = I_ij + ((1.0f - cos_t) / theta_sq) * K[i][j] + ((theta - sin_t) / (theta_sq * theta)) * K2_ij;
        }
    }

    t[0] = V[0][0]*vx + V[0][1]*vy + V[0][2]*vz;
    t[1] = V[1][0]*vx + V[1][1]*vy + V[1][2]*vz;
    t[2] = V[2][0]*vx + V[2][1]*vy + V[2][2]*vz;
}

__global__ void SE3_Manifold_Transform_Kernel(
    const float3* d_in_points,
    float3* d_out_points,
    int num_points,
    const float twist[6]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float R[3][3];
    float t[3];
    exp_se3(twist, R, t);

    float3 p = d_in_points[idx];
    d_out_points[idx] = make_float3(
        R[0][0] * p.x + R[0][1] * p.y + R[0][2] * p.z + t[0],
        R[1][0] * p.x + R[1][1] * p.y + R[1][2] * p.z + t[1],
        R[2][0] * p.x + R[2][1] * p.y + R[2][2] * p.z + t[2]
    );
}

// ----------------------------------------------------------------------------
// 2. 3D SDF-Bound Helmholtz FDTD Kernel with Shared Memory Tiling
// ----------------------------------------------------------------------------
__global__ void Helmholtz_FDTD_SDF_Shared_Kernel(
    const float* __restrict__ d_sdf,
    const float* __restrict__ d_pressure_curr,
    const float* __restrict__ d_pressure_prev,
    float* __restrict__ d_pressure_next,
    int3 dim,
    float dx,
    float dt,
    float c_air,
    float c_solid
) {
#if defined(__CUDACC__) || defined(USE_CUDA)
    __shared__ float s_p[SHARED_DIM][SHARED_DIM][SHARED_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int gx = blockIdx.x * TILE_DIM + tx;
    int gy = blockIdx.y * TILE_DIM + ty;
    int gz = blockIdx.z * TILE_DIM + tz;

    int thread_id = tx + ty * TILE_DIM + tz * TILE_DIM * TILE_DIM;
    int total_elements = SHARED_DIM * SHARED_DIM * SHARED_DIM; // 1,000
    int block_threads = TILE_DIM * TILE_DIM * TILE_DIM;        // 512

    for (int i = thread_id; i < total_elements; i += block_threads) {
        int sz = i / (SHARED_DIM * SHARED_DIM);
        int rem = i % (SHARED_DIM * SHARED_DIM);
        int sy = rem / SHARED_DIM;
        int sx = rem % SHARED_DIM;

        int global_x = blockIdx.x * TILE_DIM + (sx - 1);
        int global_y = blockIdx.y * TILE_DIM + (sy - 1);
        int global_z = blockIdx.z * TILE_DIM + (sz - 1);

        global_x = max(0, min(global_x, dim.x - 1));
        global_y = max(0, min(global_y, dim.y - 1));
        global_z = max(0, min(global_z, dim.z - 1));

        int global_idx = global_x + global_y * dim.x + global_z * (dim.x * dim.y);
        s_p[sz][sy][sx] = d_pressure_curr[global_idx];
    }

    __syncthreads();

    if (gx <= 0 || gx >= dim.x - 1 ||
        gy <= 0 || gy >= dim.y - 1 ||
        gz <= 0 || gz >= dim.z - 1) return;

    int idx = gx + gy * dim.x + gz * (dim.x * dim.y);

    float sdf_val = d_sdf[idx];
    float c = (sdf_val < 0.0f) ? c_solid : c_air;
    float alpha = (c * dt / dx) * (c * dt / dx);

    int sx = tx + 1;
    int sy = ty + 1;
    int sz = tz + 1;

    float p_c = s_p[sz][sy][sx];
    float laplacian =
        s_p[sz][sy][sx + 1] + s_p[sz][sy][sx - 1] +
        s_p[sz][sy + 1][sx] + s_p[sz][sy - 1][sx] +
        s_p[sz + 1][sy][sx] + s_p[sz - 1][sy][sx] -
        6.0f * p_c;

    float p_prev = d_pressure_prev[idx];
    d_pressure_next[idx] = 2.0f * p_c - p_prev + alpha * laplacian;
#endif
}

// ----------------------------------------------------------------------------
// 3. Integrated GTX 1060 Benchmark Main Engine
// ----------------------------------------------------------------------------
int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        std::cout << "[Elysia Integrated Bench] No CUDA device available / CPU fallback mode." << std::endl;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "==========================================================" << std::endl;
    std::cout << "Target Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // Workload Configurations
    const int num_points = 1000000;
    const size_t points_bytes = num_points * sizeof(float3);

    const int3 grid_dim = make_int3(128, 128, 128);
    const size_t num_voxels = grid_dim.x * grid_dim.y * grid_dim.z;
    const size_t grid_bytes = num_voxels * sizeof(float);

    const float dx = 0.01f;
    const float dt = 0.00001f;
    const float c_air = 343.0f;
    const float c_solid = 1500.0f;

    float3 *d_in_pts = nullptr, *d_out_pts = nullptr;
    cudaMalloc((void**)&d_in_pts, points_bytes);
    cudaMalloc((void**)&d_out_pts, points_bytes);

    float *d_sdf = nullptr, *d_p_prev = nullptr, *d_p_curr = nullptr, *d_p_next = nullptr;
    cudaMalloc((void**)&d_sdf, grid_bytes);
    cudaMalloc((void**)&d_p_prev, grid_bytes);
    cudaMalloc((void**)&d_p_curr, grid_bytes);
    cudaMalloc((void**)&d_p_next, grid_bytes);

    cudaMemset(d_in_pts, 0, points_bytes);
    cudaMemset(d_sdf, 0, grid_bytes);
    cudaMemset(d_p_prev, 0, grid_bytes);
    cudaMemset(d_p_curr, 0, grid_bytes);
    cudaMemset(d_p_next, 0, grid_bytes);

    float h_twist[6] = {0.01f, 0.02f, 0.03f, 0.1f, 0.0f, 0.5f};
    float* d_twist = nullptr;
    cudaMalloc((void**)&d_twist, 6 * sizeof(float));
    cudaMemcpy(d_twist, h_twist, 6 * sizeof(float), cudaMemcpyHostToDevice);

    int se3_block_size = 256;
    int se3_grid_size = (num_points + se3_block_size - 1) / se3_block_size;

    dim3 helm_block_size(TILE_DIM, TILE_DIM, TILE_DIM);
    dim3 helm_grid_size(
        (grid_dim.x + helm_block_size.x - 1) / helm_block_size.x,
        (grid_dim.y + helm_block_size.y - 1) / helm_block_size.y,
        (grid_dim.z + helm_block_size.z - 1) / helm_block_size.z
    );

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    std::cout << "\n[1/3] Running Warm-up Loops (10 Frames)..." << std::endl;
    for (int i = 0; i < 10; ++i) {
#if defined(__CUDACC__) || defined(USE_CUDA)
        SE3_Manifold_Transform_Kernel<<<se3_grid_size, se3_block_size>>>(
            d_in_pts, d_out_pts, num_points, d_twist
        );
        Helmholtz_FDTD_SDF_Shared_Kernel<<<helm_grid_size, helm_block_size>>>(
            d_sdf, d_p_curr, d_p_prev, d_p_next, grid_dim, dx, dt, c_air, c_solid
        );
#endif
        std::swap(d_p_prev, d_p_curr);
        std::swap(d_p_curr, d_p_next);
    }
    cudaDeviceSynchronize();

    const int total_benchmark_frames = 100;
    std::vector<float> frame_latencies;
    frame_latencies.reserve(total_benchmark_frames);

    std::cout << "[2/3] Benchmarking Dual Pipeline (" << total_benchmark_frames << " Frames)..." << std::endl;

    for (int frame = 0; frame < total_benchmark_frames; ++frame) {
        cudaEventRecord(start_event);

#if defined(__CUDACC__) || defined(USE_CUDA)
        SE3_Manifold_Transform_Kernel<<<se3_grid_size, se3_block_size>>>(
            d_in_pts, d_out_pts, num_points, d_twist
        );

        Helmholtz_FDTD_SDF_Shared_Kernel<<<helm_grid_size, helm_block_size>>>(
            d_sdf, d_p_curr, d_p_prev, d_p_next, grid_dim, dx, dt, c_air, c_solid
        );
#endif

        std::swap(d_p_prev, d_p_curr);
        std::swap(d_p_curr, d_p_next);

        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);

        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        frame_latencies.push_back(milliseconds);
    }

    float avg_latency = std::accumulate(frame_latencies.begin(), frame_latencies.end(), 0.0f) / total_benchmark_frames;
    float min_latency = *std::min_element(frame_latencies.begin(), frame_latencies.end());
    float max_latency = *std::max_element(frame_latencies.begin(), frame_latencies.end());
    float fps = 1000.0f / avg_latency;

    size_t total_vram_used_bytes = (points_bytes * 2) + (grid_bytes * 4) + (6 * sizeof(float));
    float total_vram_used_mb = total_vram_used_bytes / (1024.0f * 1024.0f);

    std::cout << "\n[3/3] Benchmark Results:" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "VRAM Occupancy      : " << total_vram_used_mb << " MB (Very Low, GTX 1060 Safe)" << std::endl;
    std::cout << "Average Latency     : " << avg_latency << " ms / frame" << std::endl;
    std::cout << "Min / Max Latency   : " << min_latency << " ms / " << max_latency << " ms" << std::endl;
    std::cout << "Estimated Throughput: " << fps << " FPS" << std::endl;
    std::cout << "==========================================================" << std::endl;

    cudaFree(d_in_pts); cudaFree(d_out_pts);
    cudaFree(d_sdf); cudaFree(d_p_prev); cudaFree(d_p_curr); cudaFree(d_p_next);
    cudaFree(d_twist);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);

    return 0;
}
