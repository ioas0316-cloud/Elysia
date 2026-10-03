#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#if defined(__CUDACC__) || defined(USE_CUDA)
#include <cuda_runtime.h>
#else
#include "elysia/cuda_host_stub.h"
#endif

#include <Eigen/Dense>

// ============================================================================
// 1. Data Structures & CUDA Hardware Buffers
// ============================================================================
struct VoxelGrid3D {
    float* d_sdf;            // GPU Device Memory: 3D SDF Array
    int*   d_labels;         // GPU Device Memory: Topology Labels (CCL)
    int3   dim;              // Grid Dimensions (e.g., 128 x 128 x 128)
    float3 voxel_size;       // Voxel Pitch (dx, dy, dz)
};

struct CutPlane {
    float3 center;           // 절단면 중심점
    float3 normal;           // 절단면 법선 벡터
};

struct TopologyFeatures {
    int beta_0;                             // 독립 개체 수 (Betti-0)
    std::vector<Eigen::Vector3f> centers;  // 개체별 질량 중심 (x_i)
    std::vector<Eigen::Matrix3f> inertia;  // 개체별 관성 텐서 (Sigma_i)
    float surface_to_volume_ratio;         // 표면적/부피 비율
};

// ============================================================================
// 2. CUDA Kernels: SDF Advection Cut & Topology Union-Find
// ============================================================================
__global__ void SDF_Advect_And_Cut_Kernel(float* d_sdf, int3 dim, float3 voxel_size, CutPlane plane) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= dim.x || y >= dim.y || z >= dim.z) return;

    int idx = x + y * dim.x + z * (dim.x * dim.y);
    float3 pos = make_float3(x * voxel_size.x, y * voxel_size.y, z * voxel_size.z);

    // CSG Smooth Subtraction plane SDF
    float dist_to_plane = (pos.x - plane.center.x) * plane.normal.x +
                          (pos.y - plane.center.y) * plane.normal.y +
                          (pos.z - plane.center.z) * plane.normal.z;
    float current_d = d_sdf[idx];

    // 절단면에 의한 SDF 지형 변형 (CSG Difference)
    d_sdf[idx] = fmaxf(current_d, -dist_to_plane);
}

__global__ void SDF_Topology_UnionFind_Kernel(const float* d_sdf, int* d_labels, int3 dim) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= dim.x || y >= dim.y || z >= dim.z) return;

    int idx = x + y * dim.x + z * (dim.x * dim.y);

    if (d_sdf[idx] < 0.0f) { // d(x) < 0 인 내부 보셀만 연산
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

// ============================================================================
// 3. Host C++ Inverse Grounding Engine
// ============================================================================
class InverseGroundingEngine {
private:
    Eigen::Matrix<float, 3, 64> W_g;    // 3 x 64 접지 투영 행렬
    Eigen::Matrix<float, 64, 3> W_g_pinv; // 64 x 3 무어-펜로즈 의사역행렬

public:
    InverseGroundingEngine() {
        W_g.setRandom(); // 실제 시스템에서는 아동기 학습으로 잠금(Lock-In)된 상수
        // 무어-펜로즈 의사역행렬 계산: W_g^+ = W_g^T * (W_g * W_g^T)^-1
        W_g_pinv = W_g.transpose() * (W_g * W_g.transpose()).inverse();
    }

    Eigen::VectorXf decodeTopologyToJamo(const TopologyFeatures& topo) {
        // Phase A: 위상 불변 특성을 3D 공간 파수(k_spatial)로 합성
        Eigen::Vector3f k_spatial = Eigen::Vector3f::Zero();

        if (topo.beta_0 > 1 && topo.centers.size() >= 2) { // 위상 분할 발생시 (Topology Split)
            k_spatial += (topo.centers[1] - topo.centers[0]).normalized() * (float)topo.beta_0;
        } else if (!topo.centers.empty()) {
            k_spatial += topo.centers[0].normalized();
        }

        // Phase B: Inverse Grounding (64차원 기호 파수 복원)
        // k_text,decoded = W_g^+ * k_spatial
        Eigen::Matrix<float, 64, 1> k_text_decoded = W_g_pinv * k_spatial;

        return k_text_decoded;
    }
};

// ============================================================================
// 4. Core Integrated Execution Loop
// ============================================================================
void ElysiaCoreExecutionStep(
    VoxelGrid3D& grid,
    const CutPlane& causal_force,
    InverseGroundingEngine& decoder
) {
    dim3 blockSize(8, 8, 8);
    dim3 gridSize(
        (grid.dim.x + blockSize.x - 1) / blockSize.x,
        (grid.dim.y + blockSize.y - 1) / blockSize.y,
        (grid.dim.z + blockSize.z - 1) / blockSize.z
    );

    // ------------------------------------------------------------------------
    // STEP 1: CUDA Causal Force Advection & SDF Cutting Kernel
    // ------------------------------------------------------------------------
#if defined(__CUDACC__) || defined(USE_CUDA)
    SDF_Advect_And_Cut_Kernel<<<gridSize, blockSize>>>(grid.d_sdf, grid.dim, grid.voxel_size, causal_force);
    cudaDeviceSynchronize();
#else
    for (int z = 0; z < grid.dim.z; ++z) {
        for (int y = 0; y < grid.dim.y; ++y) {
            for (int x = 0; x < grid.dim.x; ++x) {
                int idx = x + y * grid.dim.x + z * (grid.dim.x * grid.dim.y);
                float dist_to_plane = (x * grid.voxel_size.x - causal_force.center.x) * causal_force.normal.x +
                                      (y * grid.voxel_size.y - causal_force.center.y) * causal_force.normal.y +
                                      (z * grid.voxel_size.z - causal_force.center.z) * causal_force.normal.z;
                grid.d_sdf[idx] = std::max(grid.d_sdf[idx], -dist_to_plane);
            }
        }
    }
#endif

    // ------------------------------------------------------------------------
    // STEP 2: CUDA Parallel Topology Split Detection (Connected Components)
    // ------------------------------------------------------------------------
#if defined(__CUDACC__) || defined(USE_CUDA)
    SDF_Topology_UnionFind_Kernel<<<gridSize, blockSize>>>(grid.d_sdf, grid.d_labels, grid.dim);
    cudaDeviceSynchronize();
#endif

    // ------------------------------------------------------------------------
    // STEP 3: GPU Memory Reduction & Feature Extraction (Host Memory Transfer)
    // ------------------------------------------------------------------------
    std::vector<int> h_labels(grid.dim.x * grid.dim.y * grid.dim.z);
    std::vector<float> h_sdf(grid.dim.x * grid.dim.y * grid.dim.z);

    cudaMemcpy(h_labels.data(), grid.d_labels, h_labels.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sdf.data(), grid.d_sdf, h_sdf.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Betti-0 및 질량중심 수집 (Host Reduction)
    TopologyFeatures topo;
    std::vector<int> unique_components;
    for (size_t i = 0; i < h_sdf.size(); ++i) {
        if (h_sdf[i] < 0.0f) {
            int lbl = h_labels[i];
            if (std::find(unique_components.begin(), unique_components.end(), lbl) == unique_components.end()) {
                unique_components.push_back(lbl);
            }
        }
    }
    topo.beta_0 = unique_components.size();

    // ------------------------------------------------------------------------
    // STEP 4: Inverse Grounding & Re-encoding (SDF Morph -> Jamo Wavevector)
    // ------------------------------------------------------------------------
    Eigen::VectorXf k_text_decoded = decoder.decodeTopologyToJamo(topo);

    // ------------------------------------------------------------------------
    // STEP 5: Execution Loop Log & Decision
    // ------------------------------------------------------------------------
    std::cout << "[Elysia Core Loop] SDF Topology Split Executed." << std::endl;
    std::cout << "  └─ Detected Betti-0 (beta_0): " << topo.beta_0 << " object(s)" << std::endl;
    std::cout << "  └─ Decoded Wavevector L2-Norm: " << k_text_decoded.norm() << std::endl;

    if (topo.beta_0 > 1) {
        std::cout << "  └─ [Causal Re-encoding]: Symbol '두 조각으로 분할됨' Matched!" << std::endl;
    }
}

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        std::cout << "[Elysia Core Bench] No CUDA device available / CPU fallback mode." << std::endl;
    }

    int3 dim = make_int3(64, 64, 64);
    size_t num_voxels = dim.x * dim.y * dim.z;
    float3 voxel_size = make_float3(0.01f, 0.01f, 0.01f);

    VoxelGrid3D grid;
    grid.dim = dim;
    grid.voxel_size = voxel_size;

    cudaMalloc((void**)&grid.d_sdf, num_voxels * sizeof(float));
    cudaMalloc((void**)&grid.d_labels, num_voxels * sizeof(int));

    std::vector<float> h_sdf(num_voxels, 1.0f);
    std::vector<int> h_labels(num_voxels);

    for (int z = 0; z < dim.z; ++z) {
        for (int y = 0; y < dim.y; ++y) {
            for (int x = 0; x < dim.x; ++x) {
                int idx = x + y * dim.x + z * (dim.x * dim.y);
                h_labels[idx] = idx;
                float dx = (x - 32) * 0.01f;
                float dy = (y - 32) * 0.01f;
                float dz = (z - 32) * 0.01f;
                float dist = std::sqrt(dx*dx + dy*dy + dz*dz) - 0.2f;
                h_sdf[idx] = dist;
            }
        }
    }

    cudaMemcpy(grid.d_sdf, h_sdf.data(), num_voxels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grid.d_labels, h_labels.data(), num_voxels * sizeof(int), cudaMemcpyHostToDevice);

    CutPlane cut = { make_float3(0.32f, 0.32f, 0.32f), make_float3(0.0f, 1.0f, 0.0f) };
    InverseGroundingEngine decoder;

    std::cout << "=== Elysia Core Execution Loop Test ===" << std::endl;
    ElysiaCoreExecutionStep(grid, cut, decoder);

    cudaFree(grid.d_sdf);
    cudaFree(grid.d_labels);
    return 0;
}
