#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#else
#include "elysia/cuda_host_stub.h"
#include <cmath>
#endif

extern "C" __global__ void k_parasympathetic_node_fusion(
    const float2* __restrict__ child1_rotors,   // 자식 노드 1 위상
    const float2* __restrict__ child2_rotors,   // 자식 노드 2 위상
    float2*       __restrict__ fused_rotors,    // 융합될 모체 노드 위상
    int*          __restrict__ fusion_mask,     // 융합 성공 여부 마스크 (1 = 융합)
    const float   coherence_threshold,          // 융합을 위한 최소 위상 결맞음 기준
    const int     num_split_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_split_pairs) return;

    float2 c1 = child1_rotors[idx];
    float2 c2 = child2_rotors[idx];

    float mag1_sq = c1.x * c1.x + c1.y * c1.y;
    float mag2_sq = c2.x * c2.x + c2.y * c2.y;

    if (mag1_sq < 1e-7f || mag2_sq < 1e-7f) {
        fusion_mask[idx] = 0;
        return;
    }

    float dot_product = c1.x * c2.x + c1.y * c2.y;
    float phase_coherence = dot_product / sqrtf(mag1_sq * mag2_sq);

    if (phase_coherence >= coherence_threshold) {
        float sum_x = c1.x + c2.x;
        float sum_y = c1.y + c2.y;
        float sum_mag = sqrtf(sum_x * sum_x + sum_y * sum_y);

        if (sum_mag > 1e-7f) {
            fused_rotors[idx] = make_float2(sum_x / sum_mag, sum_y / sum_mag);
            fusion_mask[idx] = 1;
        } else {
            fusion_mask[idx] = 0;
        }
    } else {
        fusion_mask[idx] = 0;
    }
}
