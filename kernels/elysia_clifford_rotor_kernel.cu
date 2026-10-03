#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// ============================================================================
// Quaternion Struct & Rotor Operations in Clifford Algebra
// ============================================================================
struct QuaternionRotor {
    float w, x, y, z; // q = w + x*i + y*j + z*k
};

// 쿼터니언 로터 기반 3D 공간 벡터 회전: x' = q * x * q*
__device__ inline float3 rotate_vector_by_rotor(float3 v, QuaternionRotor q) {
    // t = 2 * cross(q.xyz, v)
    float tx = 2.0f * (q.y * v.z - q.z * v.y);
    float ty = 2.0f * (q.z * v.x - q.x * v.z);
    float tz = 2.0f * (q.x * v.y - q.y * v.x);

    // v' = v + q.w * t + cross(q.xyz, t)
    float3 v_rot;
    v_rot.x = v.x + q.w * tx + (q.y * tz - q.z * ty);
    v_rot.y = v.y + q.w * ty + (q.z * tx - q.x * tz);
    v_rot.z = v.z + q.w * tz + (q.x * ty - q.y * tx);

    return v_rot;
}

// ============================================================================
// CUDA Kernel: Clifford Quaternion Rotor Accelerated SDF Resonance
// ============================================================================
__global__ void clifford_rotor_sdf_kernel(
    const float* __restrict__ query_pos,  // [N_points, 3]
    const float* __restrict__ k_wave,     // [3]
    const QuaternionRotor rotor,          // 4D Quaternion Rotor
    const float omega_t,
    const float amplitude,
    const float sigma,
    float* __restrict__ d_out,            // Output Distance
    float* __restrict__ normal_out,       // Output Analytic Normal
    const int N_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_points) return;

    // 1. Raw Point Coordinates
    float3 pos = make_float3(
        query_pos[idx * 3 + 0],
        query_pos[idx * 3 + 1],
        query_pos[idx * 3 + 2]
    );

    // 2. Apply Clifford Quaternion Rotor Rotation (Zero-Matrix Register Rotation)
    float3 rotated_pos = rotate_vector_by_rotor(pos, rotor);

    // 3. Base Sphere Distance in Rotated Frame
    float norm_x = sqrtf(rotated_pos.x * rotated_pos.x +
                         rotated_pos.y * rotated_pos.y +
                         rotated_pos.z * rotated_pos.z + 1e-8f);
    float d_base = norm_x - 1.0f;

    // 4. Phase Interference: φ = k · x_rot - ωt
    float phase = (k_wave[0] * rotated_pos.x +
                   k_wave[1] * rotated_pos.y +
                   k_wave[2] * rotated_pos.z) - omega_t;

    float cos_p = cosf(phase);
    float sin_p = sinf(phase);
    float mask = expf(-fabsf(d_base) / sigma);

    // Distance Output
    d_out[idx] = d_base + amplitude * mask * cos_p;

    // 5. Analytical Gradient in Rotated Frame
    float grad_base_x = rotated_pos.x / norm_x;
    float grad_base_y = rotated_pos.y / norm_x;
    float grad_base_z = rotated_pos.z / norm_x;

    float sgn_d = (d_base > 0.0f) ? 1.0f : ((d_base < 0.0f) ? -1.0f : 0.0f);
    float d_mask_term = -(sgn_d / sigma) * mask * cos_p;
    float d_phase_term = -mask * sin_p;

    float3 rot_normal;
    rot_normal.x = grad_base_x * (1.0f + amplitude * d_mask_term) + k_wave[0] * (amplitude * d_phase_term);
    rot_normal.y = grad_base_y * (1.0f + amplitude * d_mask_term) + k_wave[1] * (amplitude * d_phase_term);
    rot_normal.z = grad_base_z * (1.0f + amplitude * d_mask_term) + k_wave[2] * (amplitude * d_phase_term);

    // Inverse Rotation on Normal Vector (세계 좌표계로 Normal 복원)
    QuaternionRotor inv_rotor = { rotor.w, -rotor.x, -rotor.y, -rotor.z };
    float3 world_normal = rotate_vector_by_rotor(rot_normal, inv_rotor);

    // Normal Normalization
    float n_len = sqrtf(world_normal.x * world_normal.x +
                        world_normal.y * world_normal.y +
                        world_normal.z * world_normal.z + 1e-8f);

    normal_out[idx * 3 + 0] = world_normal.x / n_len;
    normal_out[idx * 3 + 1] = world_normal.y / n_len;
    normal_out[idx * 3 + 2] = world_normal.z / n_len;
}
