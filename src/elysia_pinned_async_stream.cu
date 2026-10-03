#include <iostream>
#include <vector>

#if defined(__CUDACC__) || defined(USE_CUDA)
#include <cuda_runtime.h>
#else
#include "elysia/cuda_host_stub.h"
#endif

#include <Eigen/Dense>

#define NUM_STREAMS 2  // 더블 버퍼링용 2개 병렬 비동기 스트림

struct CutPlane {
    float3 center;
    float3 normal;
};

// ----------------------------------------------------------------------------
// CUDA Kernels (Stream Aware)
// ----------------------------------------------------------------------------
__global__ void SDF_Advect_And_Cut_Kernel(float* d_sdf, int3 dim, float3 voxel_size, CutPlane plane) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= dim.x || y >= dim.y || z >= dim.z) return;

    int idx = x + y * dim.x + z * (dim.x * dim.y);
    float3 pos = make_float3(x * voxel_size.x, y * voxel_size.y, z * voxel_size.z);

    float dist_to_plane = (pos.x - plane.center.x) * plane.normal.x +
                          (pos.y - plane.center.y) * plane.normal.y +
                          (pos.z - plane.center.z) * plane.normal.z;
    d_sdf[idx] = fmaxf(d_sdf[idx], -dist_to_plane);
}

__global__ void SDF_Topology_UnionFind_Kernel(const float* d_sdf, int* d_labels, int3 dim) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= dim.x || y >= dim.y || z >= dim.z) return;

    int idx = x + y * dim.x + z * (dim.x * dim.y);

    if (d_sdf[idx] < 0.0f) {
        int offsets[6] = {-1, 1, -dim.x, dim.x, -(dim.x * dim.y), (dim.x * dim.y)};
        for (int i = 0; i < 6; ++i) {
            int n_idx = idx + offsets[i];
            if (n_idx >= 0 && n_idx < (dim.x * dim.y * dim.z) && d_sdf[n_idx] < 0.0f) {
#if defined(__CUDACC__) || defined(USE_CUDA)
                atomicMin(&d_labels[idx], d_labels[n_idx]);
                atomicMin(&d_labels[n_idx], d_labels[idx]);
#else
                if (d_labels[n_idx] < d_labels[idx]) d_labels[idx] = d_labels[n_idx];
#endif
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Pinned Memory & Async Stream Engine Class
// ----------------------------------------------------------------------------
class AsyncElysiaStreamEngine {
private:
    int3 dim;
    size_t num_voxels;
    size_t sdf_bytes;
    size_t label_bytes;

    // GPU Device Buffers (2 Set for Double Buffering)
    float* d_sdf[NUM_STREAMS];
    int*   d_labels[NUM_STREAMS];

    // CPU Pinned Host Buffers (cudaMallocHost Page-Locked Memory)
    float* h_pinned_sdf[NUM_STREAMS];
    int*   h_pinned_labels[NUM_STREAMS];

    // CUDA Non-blocking Streams & Events
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t  start_events[NUM_STREAMS];
    cudaEvent_t  stop_events[NUM_STREAMS];

public:
    AsyncElysiaStreamEngine(int3 grid_dim) : dim(grid_dim) {
        num_voxels = dim.x * dim.y * dim.z;
        sdf_bytes = num_voxels * sizeof(float);
        label_bytes = num_voxels * sizeof(int);

        for (int i = 0; i < NUM_STREAMS; ++i) {
            // 1. CUDA Async Stream 생성 (Non-blocking)
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
            cudaEventCreate(&start_events[i]);
            cudaEventCreate(&stop_events[i]);

            // 2. Host Page-Locked (Pinned) Memory 할당 -> DMA direct transfer 가속
            cudaMallocHost((void**)&h_pinned_sdf[i], sdf_bytes);
            cudaMallocHost((void**)&h_pinned_labels[i], label_bytes);

            // 3. Device VRAM Buffers 할당
            cudaMalloc((void**)&d_sdf[i], sdf_bytes);
            cudaMalloc((void**)&d_labels[i], label_bytes);

            // 초기 보셀 필드 가공 (예시: 초기 구체 형태 SDF)
            for (size_t v = 0; v < num_voxels; ++v) {
                h_pinned_sdf[i][v] = -1.0f; // Inside surface
                h_pinned_labels[i][v] = static_cast<int>(v);
            }
        }
    }

    ~AsyncElysiaStreamEngine() {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            cudaFreeHost(h_pinned_sdf[i]);
            cudaFreeHost(h_pinned_labels[i]);
            cudaFree(d_sdf[i]);
            cudaFree(d_labels[i]);

            cudaStreamDestroy(streams[i]);
            cudaEventDestroy(start_events[i]);
            cudaEventDestroy(stop_events[i]);
        }
    }

    // ------------------------------------------------------------------------
    // Pinned Async Overlapped Core Pipeline Step
    // ------------------------------------------------------------------------
    void ExecutePipelinedFrame(const std::vector<CutPlane>& causal_cuts) {
        dim3 blockSize(8, 8, 8);
        dim3 gridSize(
            (dim.x + blockSize.x - 1) / blockSize.x,
            (dim.y + blockSize.y - 1) / blockSize.y,
            (dim.z + blockSize.z - 1) / blockSize.z
        );

        float3 voxel_size = make_float3(0.01f, 0.01f, 0.01f);

        // 스트림별 비동기 파이프라인 구동 (Stream 0 & Stream 1 Overlapping)
        for (int s = 0; s < NUM_STREAMS; ++s) {
            cudaEventRecord(start_events[s], streams[s]);

            // Phase 1: Host -> Device Async Copy (PCIe DMA 가속)
            cudaMemcpyAsync(d_sdf[s], h_pinned_sdf[s], sdf_bytes, cudaMemcpyHostToDevice, streams[s]);
            cudaMemcpyAsync(d_labels[s], h_pinned_labels[s], label_bytes, cudaMemcpyHostToDevice, streams[s]);

            // Phase 2: Kernel Executions bound to Stream[s]
#if defined(__CUDACC__) || defined(USE_CUDA)
            SDF_Advect_And_Cut_Kernel<<<gridSize, blockSize, 0, streams[s]>>>(
                d_sdf[s], dim, voxel_size, causal_cuts[s]
            );

            SDF_Topology_UnionFind_Kernel<<<gridSize, blockSize, 0, streams[s]>>>(
                d_sdf[s], d_labels[s], dim
            );
#endif

            // Phase 3: Device -> Host Async Copy (비동기 결과 회수)
            cudaMemcpyAsync(h_pinned_labels[s], d_labels[s], label_bytes, cudaMemcpyDeviceToHost, streams[s]);

            cudaEventRecord(stop_events[s], streams[s]);
        }

        // 전체 스트림 동기화 및 프로파일링
        for (int s = 0; s < NUM_STREAMS; ++s) {
            cudaStreamSynchronize(streams[s]);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start_events[s], stop_events[s]);
            std::cout << "[Stream " << s << "] Pipelined Step Executed in: " << ms << " ms" << std::endl;
        }
    }
};

// ----------------------------------------------------------------------------
// Main Execution Entry
// ----------------------------------------------------------------------------
int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        std::cout << "[Elysia Pinned Async Stream] No CUDA device available / CPU fallback mode." << std::endl;
    }

    int3 grid_dim = make_int3(64, 64, 64);
    AsyncElysiaStreamEngine engine(grid_dim);

    std::vector<CutPlane> cuts(NUM_STREAMS);
    cuts[0] = { make_float3(0.32f, 0.32f, 0.32f), make_float3(0.0f, 1.0f, 0.0f) };
    cuts[1] = { make_float3(0.25f, 0.25f, 0.25f), make_float3(1.0f, 0.0f, 0.0f) };

    std::cout << "=== GTX 1060 Optimized Async Pinned Stream Loop Start ===" << std::endl;
    for (int frame = 0; frame < 5; ++frame) {
        std::cout << "\n--- Frame " << frame + 1 << " ---" << std::endl;
        engine.ExecutePipelinedFrame(cuts);
    }

    return 0;
}
