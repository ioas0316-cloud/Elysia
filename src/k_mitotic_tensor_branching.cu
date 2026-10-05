#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#else
#include "elysia/cuda_host_stub.h"
#include <cmath>
#endif

extern "C" __global__ void k_mitotic_tensor_branching(
    const float2* __restrict__ parent_rotors,      // 모체 관측 위상 (N)
    const float*  __restrict__ curvature_map,       // 공간 곡률 지도 (N)
    float2*       __restrict__ child1_rotors,       // 직교 위상 자식 1 (N)
    float2*       __restrict__ child2_rotors,       // 직교 위상 자식 2 (N)
    int*          __restrict__ branch_mask,         // 분열 발생 유무 마스크 (N)
    const float   curvature_threshold,              // 분열 유도 임계 곡률
    const int     num_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    float cur = curvature_map[idx];

    if (cur >= curvature_threshold) {
        float2 p = parent_rotors[idx];

        const float norm_factor = 0.70710678f;
        const float cos_p = 0.70710678f;
        const float sin_p = 0.70710678f;

        float2 c1;
        c1.x = (p.x * cos_p - p.y * sin_p) * norm_factor;
        c1.y = (p.x * sin_p + p.y * cos_p) * norm_factor;

        float2 c2;
        c2.x = (p.x * cos_p + p.y * sin_p) * norm_factor;
        c2.y = (-p.x * sin_p + p.y * cos_p) * norm_factor;

        child1_rotors[idx] = c1;
        child2_rotors[idx] = c2;
        branch_mask[idx]   = 1;
    } else {
        child1_rotors[idx] = parent_rotors[idx];
        child2_rotors[idx] = make_float2(0.0f, 0.0f);
        branch_mask[idx]   = 0;
    }
}
