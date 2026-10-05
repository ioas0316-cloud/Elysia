#include "elysia_autonomic_controller.hpp"
#include <cmath>
#include <algorithm>

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <math_constants.h>
#include <device_launch_parameters.h>

__global__ void k_parasympathetic_consolidation(
    float2* __restrict__ rotors,
    float*  __restrict__ coherence_map,
    const float ach_level,
    const float noise_prune_threshold,
    const int num_rotors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rotors) return;

    float2 r = rotors[idx];
    float mag_sq = r.x * r.x + r.y * r.y;
    float mag = sqrtf(mag_sq);

    float dynamic_threshold = noise_prune_threshold * (1.0f + ach_level);
    if (mag < dynamic_threshold) {
        rotors[idx] = make_float2(0.0f, 0.0f);
        coherence_map[idx] = 0.0f;
        return;
    }

    if (mag > 1e-7f) {
        float inv_mag = 1.0f / mag;
        float target_mag = 1.0f;
        float relaxation_rate = 0.20f * (1.0f - ach_level);

        float new_mag = mag + relaxation_rate * (target_mag - mag);
        rotors[idx] = make_float2((r.x * inv_mag) * new_mag, (r.y * inv_mag) * new_mag);
        coherence_map[idx] = new_mag / target_mag;
    }
}

#endif // __CUDACC__

extern "C" {

void launch_parasympathetic_consolidation_kernel(
    float2* d_rotors,
    float* d_coherence_map,
    float ach_level,
    float noise_prune_threshold,
    int num_rotors,
    cudaStream_t stream)
{
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
    int block_size = 256;
    int grid_size = (num_rotors + block_size - 1) / block_size;
    k_parasympathetic_consolidation<<<grid_size, block_size, 0, stream>>>(
        d_rotors, d_coherence_map, ach_level, noise_prune_threshold, num_rotors
    );
#else
    for (int idx = 0; idx < num_rotors; ++idx) {
        float2 r = d_rotors[idx];
        float mag = std::sqrt(r.x * r.x + r.y * r.y);
        float dynamic_threshold = noise_prune_threshold * (1.0f + ach_level);

        if (mag < dynamic_threshold) {
            d_rotors[idx] = make_float2(0.0f, 0.0f);
            d_coherence_map[idx] = 0.0f;
        } else if (mag > 1e-7f) {
            float inv_mag = 1.0f / mag;
            float target_mag = 1.0f;
            float relaxation_rate = 0.20f * (1.0f - ach_level);
            float new_mag = mag + relaxation_rate * (target_mag - mag);
            d_rotors[idx] = make_float2((r.x * inv_mag) * new_mag, (r.y * inv_mag) * new_mag);
            d_coherence_map[idx] = new_mag / target_mag;
        }
    }
#endif
}

} // extern "C"
