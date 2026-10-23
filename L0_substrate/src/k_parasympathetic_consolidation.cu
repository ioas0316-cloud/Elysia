#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#else
#include "elysia/cuda_host_stub.h"
#include <cmath>
#endif

extern "C" __global__ void k_parasympathetic_consolidation(
    float2* __restrict__ rotors,        // complex float representation (cos θ, sin θ)
    float* __restrict__ coherence_map,  // 각 로터의 위상 안정도
    const float ach_level,              // 컨트롤러가 전달한 ACh 농도
    const float noise_prune_threshold,  // 가지치기 미세 오차 임계값
    const int num_rotors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rotors) return;

    float2 r = rotors[idx];
    float mag_sq = r.x * r.x + r.y * r.y;
    float mag = sqrtf(mag_sq);

    // 1. 노이즈 가지치기 (Pruning): 진폭이 임계값 이하인 유동 위상은 0으로 수축
    float dynamic_threshold = noise_prune_threshold * (1.0f + ach_level);
    if (mag < dynamic_threshold) {
        rotors[idx] = make_float2(0.0f, 0.0f);
        coherence_map[idx] = 0.0f;
        return;
    }

    // 2. 단축 규격화 (Fast Normalization & Phase-Locking)
    if (mag > 1e-7f) {
        float inv_mag = 1.0f / mag;
        float target_mag = 1.0f;
        float relaxation_rate = 0.20f * (1.0f - ach_level); // ACh 소진 시 고착 속도 증가

        float new_mag = mag + relaxation_rate * (target_mag - mag);
        rotors[idx] = make_float2((r.x * inv_mag) * new_mag, (r.y * inv_mag) * new_mag);
        coherence_map[idx] = new_mag / target_mag;
    }
}
