#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct __align__(16) QuaternionPhasor {
    float4 q;        // (w, x, y, z) - Quaternion Phase Angle
    float4 pos_amp;  // (x, y, z, Amplitude)
    float4 scale;    // (sx, sy, sz, Cutoff_Radius)
};

__device__ __forceinline__ float4 quaternion_mul_cuda(const float4& a, const float4& b) {
    return make_float4(
        a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w, // w
        a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z, // x
        a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y, // y
        a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x  // z
    );
}

__global__ void update_quaternion_phase_kernel(
    QuaternionPhasor* __restrict__ phasors,
    const float4* __restrict__ target_field,
    float dt, float gamma, int num_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    QuaternionPhasor node = phasors[idx];
    float4 target_q = target_field[idx];

    // Conjugate quaternion: q_inv = (w, -x, -y, -z)
    float4 q_inv = make_float4(node.q.x, -node.q.y, -node.q.z, -node.q.w);

    // Error quaternion e = target_q x q_inv
    float4 err_q = quaternion_mul_cuda(target_q, q_inv);

    // Torque vector
    float sign_w = (err_q.x >= 0.0f) ? 1.0f : -1.0f;
    float3 torque = make_float3(
        2.0f * sign_w * err_q.y,
        2.0f * sign_w * err_q.z,
        2.0f * sign_w * err_q.w
    );

    // Exponent map rotation integration
    float4 dq = quaternion_mul_cuda(make_float4(0.0f, torque.x, torque.y, torque.z), node.q);
    node.q.x += gamma * dt * 0.5f * dq.x;
    node.q.y += gamma * dt * 0.5f * dq.y;
    node.q.z += gamma * dt * 0.5f * dq.z;
    node.q.w += gamma * dt * 0.5f * dq.w;

    // Fast Inverse Square Root (rsqrtf) normalization
    float norm = rsqrtf(node.q.x*node.q.x + node.q.y*node.q.y + node.q.z*node.q.z + node.q.w*node.q.w + 1e-8f);
    node.q.x *= norm; node.q.y *= norm; node.q.z *= norm; node.q.w *= norm;

    phasors[idx] = node;
}
