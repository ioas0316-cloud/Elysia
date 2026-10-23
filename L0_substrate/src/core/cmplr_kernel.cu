#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>
#include <cmath>
#include "elysia/core/cmplr_kernel.cuh"

// 128-bit Vectorized Loader
__device__ __forceinline__ void load_multivector(const float* __restrict__ src, float4& v0, float4& v1) {
    const float4* ptr = reinterpret_cast<const float4*>(src);
    v0 = __ldcg(&ptr[0]); // L2 cache hint load
    v1 = __ldcg(&ptr[1]);
}

// Spin(3) Gauge Projection (Fast Inverse Square Root)
__device__ __forceinline__ void project_spin_device(float4& v0, float4& v1) {
    // Scalar Norm Squared using Clifford Reversion: <Psi * ~Psi>_0 = v0.x^2 + v0.y^2 + v0.z^2 + v0.w^2 - (v1.x^2 + v1.y^2 + v1.z^2 + v1.w^2)
    float norm_sq = (v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w)
                  - (v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w);

    float inv_norm = rsqrtf(fabsf(norm_sq) + 1e-8f);

    v0.x *= inv_norm; v0.y *= inv_norm; v0.z *= inv_norm; v0.w *= inv_norm;
    v1.x *= inv_norm; v1.y *= inv_norm; v1.z *= inv_norm; v1.w *= inv_norm;
}

// CMPLR Main CUDA Kernel
__global__ void cmplr_relaxation_kernel(
    float* __restrict__ Psi,                  // [N, 8]
    const int* __restrict__ row_ptr,          // [N + 1]
    const int* __restrict__ col_ind,          // [E]
    const float* __restrict__ K_tensors,      // [E, 9]
    const int N,
    const float dt
) {
    // Warp and Node ID calculation
    const int warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane_id = threadIdx.x; // 0..31
    const int node_a = warp_id;

    if (node_a >= N) return;

    // 1. Load node a state tensor into registers (Vectorized)
    float4 a_v0, a_v1;
    load_multivector(&Psi[node_a * 8], a_v0, a_v1);

    // Reverse ~Psi_a calculation (Grade 2, 3 sign inversion)
    float4 rev_a_v0 = a_v0;
    float4 rev_a_v1 = make_float4(-a_v1.x, -a_v1.y, -a_v1.z, -a_v1.w);

    // 2. Neighbor loop and bivector phase difference calculation (Warp Parallel)
    const int edge_start = row_ptr[node_a];
    const int edge_end   = row_ptr[node_a + 1];

    float local_torque_12 = 0.0f;
    float local_torque_23 = 0.0f;
    float local_torque_31 = 0.0f;

    for (int e = edge_start + lane_id; e < edge_end; e += 32) {
        int node_b = col_ind[e];

        float4 b_v0, b_v1;
        load_multivector(&Psi[node_b * 8], b_v0, b_v1);

        // Grade-2 Bivector Cross Extraction: <Psi_b * ~Psi_a>_2
        float e12 = (b_v0.x * rev_a_v1.x + b_v1.x * rev_a_v0.x) + (b_v0.y * rev_a_v0.z - b_v0.z * rev_a_v0.y) - (b_v1.y * rev_a_v1.z - b_v1.z * rev_a_v1.y);
        float e23 = (b_v0.x * rev_a_v1.y + b_v1.y * rev_a_v0.x) + (b_v0.z * rev_a_v0.w - b_v0.w * rev_a_v0.z) - (b_v1.z * rev_a_v1.x - b_v1.x * rev_a_v1.z);
        float e31 = (b_v0.x * rev_a_v1.z + b_v1.z * rev_a_v0.x) + (b_v0.w * rev_a_v0.y - b_v0.y * rev_a_v0.w) - (b_v1.x * rev_a_v1.y - b_v1.y * rev_a_v1.x);

        // K_ab Matrix Contraction (3x3 Flattened)
        const float* K = &K_tensors[e * 9];
        local_torque_12 += K[0] * e12 + K[1] * e23 + K[2] * e31;
        local_torque_23 += K[3] * e12 + K[4] * e23 + K[5] * e31;
        local_torque_31 += K[6] * e12 + K[7] * e23 + K[8] * e31;
    }

    // 3. Warp Shuffle Reduction (32 Lanes -> Lane 0)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_torque_12 += __shfl_down_sync(0xffffffff, local_torque_12, offset);
        local_torque_23 += __shfl_down_sync(0xffffffff, local_torque_23, offset);
        local_torque_31 += __shfl_down_sync(0xffffffff, local_torque_31, offset);
    }

    // 4. Lane 0: Exponential Map Rotor Update & Storage Writeback
    if (lane_id == 0) {
        float theta = sqrtf(local_torque_12 * local_torque_12 +
                            local_torque_23 * local_torque_23 +
                            local_torque_31 * local_torque_31) + 1e-8f;

        float half_dt_theta = 0.5f * dt * theta;
        float cos_val = __cosf(half_dt_theta);
        float sin_val = __sinf(half_dt_theta) / theta;

        // Bivector Generator B_a = (b12, b23, b31)
        float b12 = local_torque_12 * sin_val;
        float b23 = local_torque_23 * sin_val;
        float b31 = local_torque_31 * sin_val;

        // Rotor Multiplication: Psi_next = (cos + B_hat * sin) * Psi_a
        float4 next_v0, next_v1;
        next_v0.x = cos_val * a_v0.x - (b12 * a_v1.x + b23 * a_v1.y + b31 * a_v1.z);
        next_v0.y = cos_val * a_v0.y + (b12 * a_v0.z - b31 * a_v0.w);
        next_v0.z = cos_val * a_v0.z + (b23 * a_v0.w - b12 * a_v0.y);
        next_v0.w = cos_val * a_v0.w + (b31 * a_v0.y - b23 * a_v0.z);

        next_v1.x = cos_val * a_v1.x + (b12 * a_v0.x);
        next_v1.y = cos_val * a_v1.y + (b23 * a_v0.x);
        next_v1.z = cos_val * a_v1.z + (b31 * a_v0.x);
        next_v1.w = cos_val * a_v1.w;

        // Gauge Locking Projection
        project_spin_device(next_v0, next_v1);

        // Vectorized Writeback
        float4* dst_ptr = reinterpret_cast<float4*>(&Psi[node_a * 8]);
        dst_ptr[0] = next_v0;
        dst_ptr[1] = next_v1;
    }
}

void cmplr_step_cuda(
    torch::Tensor Psi,
    torch::Tensor row_ptr,
    torch::Tensor col_ind,
    torch::Tensor K_tensors,
    float dt
) {
    // Tensor Constraints Check
    TORCH_CHECK(Psi.is_cuda(), "Psi must be a CUDA tensor");
    TORCH_CHECK(Psi.is_contiguous(), "Psi must be contiguous");
    TORCH_CHECK(Psi.size(1) == 8, "Psi shape must be [N, 8]");

    const int N = Psi.size(0);
    dim3 blockDim(32, 8, 1); // 256 threads per block
    dim3 gridDim((N + 7) / 8, 1, 1);

    cmplr_relaxation_kernel<<<gridDim, blockDim>>>(
        Psi.data_ptr<float>(),
        row_ptr.data_ptr<int>(),
        col_ind.data_ptr<int>(),
        K_tensors.data_ptr<float>(),
        N,
        dt
    );
}
