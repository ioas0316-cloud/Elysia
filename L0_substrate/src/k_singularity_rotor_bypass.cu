#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#else
#include "elysia/cuda_host_stub.h"
#include <cmath>
using std::isnan;
using std::isinf;
#endif

extern "C" __global__ void k_singularity_rotor_bypass(
    float2* __restrict__ rotors,        // complex / bivector e12 representation
    float*  __restrict__ curvature_map, // 특이점 우회로 인한 공간 곡률 지도
    const float bypass_angle,           // 우회 회전 각도 (θ, 예: π/2)
    const int num_rotors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rotors) return;

    float2 r = rotors[idx];

    // 1. 특이점 (NaN 또는 Inf) 수치적 검출
    bool is_singular = isnan(r.x) || isnan(r.y) || isinf(r.x) || isinf(r.y);

    if (is_singular) {
        float half_angle = bypass_angle * 0.5f;
        float cos_half = cosf(half_angle);
        float sin_half = sinf(half_angle);

        float2 base_rotor = make_float2(1.0f, 0.0f);

        float new_x = base_rotor.x * cos_half - base_rotor.y * sin_half;
        float new_y = base_rotor.x * sin_half + base_rotor.y * cos_half;

        rotors[idx] = make_float2(new_x, new_y);
        curvature_map[idx] = 1.0f;
    } else {
        curvature_map[idx] = 0.0f;
    }
}
