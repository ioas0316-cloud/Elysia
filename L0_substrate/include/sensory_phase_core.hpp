#ifndef SENSORY_PHASE_CORE_HPP
#define SENSORY_PHASE_CORE_HPP

#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>

#if defined(__CUDACC__) || defined(WITH_CUDA)
#include <cuda_runtime.h>
#endif

namespace Elysia {

#ifndef __CUDACC__
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif
#endif

// Vector helper types for C++ host when CUDA headers are not natively included
#if !defined(__CUDACC__) && !defined(CUDA_VERSION)
struct alignas(16) float4 {
    float x, y, z, w;
};
struct alignas(16) uint4 {
    uint32_t x, y, z, w;
};
struct float3 {
    float x, y, z;
};
#endif

// 16-Byte Aligned Sensory Stream Input Frame
struct alignas(16) SensoryStreamInput {
    float4 position_tension;   // x, y, z: spatial coords (x_i) / w: intent tension τ
    float4 velocity_dtension;  // x, y, z: velocity v_i / w: tension change rate dτ/dt
    float4 audio_spectrum;     // x: A_L, y: A_M, z: A_H, w: base frequency ω_0
    float4 acceleration_grad;  // x, y, z: acceleration a_i / w: audio grad magnitude |∇A_M|
};

// 16-Byte Aligned Output Spacetime Metric Node
struct alignas(16) SpacetimeNodeBoundary {
    float4 metric_diag;        // M_xx, M_yy, M_zz, time dilation γ_i
    float4 metric_offdiag;     // M_xy, M_xz, M_yz, phase coherence Φ_i
    uint4  csr_topology;       // CSR adjacent node indices / phase coupling edges
};

// 16-Byte Aligned Diagnostic Tensor Field Output
struct alignas(16) TensorFieldDiagnostics {
    float4 position_gdi;      // x, y, z: position / w: Geodesic Divergence Indicator (GDI)
    float4 saddle_hessian;    // x: Hessian Det, y: Max Eig, z: Min Eig, w: Phase Lock (Φ_i)
    uint4  classification_st; // x: Phase State (0:Gas, 1:Liquid, 2:Solid, 3:Shear), y: Attractor ID, z, w: Reserved
};

// Hyperparameters for Sensory-to-Metric Mapping
struct SensoryPhaseConfig {
    float lambda_1 = 1.0f;     // Tension fluctuation penalty
    float lambda_2 = 0.5f;     // High frequency audio noise penalty
    float lambda_3 = 1.0f;     // Time dilation acceleration weight
    float kappa_1  = 1.0f;     // Isotropic expansion coefficient
    float alpha    = 0.5f;     // Kinematic velocity anisotropy coefficient
    float beta     = 0.3f;     // Audio frequency directional distortion coefficient
    float gamma_max = 3.0f;    // Max time dilation factor
    float phi_liquid = 0.4f;   // Gas -> Liquid phase transition threshold
    float phi_solid  = 0.8f;   // Liquid -> Solid phase transition threshold
    float tau_shear  = 2.0f;   // Shear breakdown threshold for dτ/dt
};

// Helper device functions
__host__ __device__ inline float calculate_phase_coherence(
    float3 v, float3 gradA_M, float dtension, float A_H,
    float lambda1, float lambda2)
{
    float v_len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    float g_len = sqrtf(gradA_M.x * gradA_M.x + gradA_M.y * gradA_M.y + gradA_M.z * gradA_M.z);
    float alignment = 1.0f;
    if (v_len > 1e-6f && g_len > 1e-6f) {
        float dot = (v.x * gradA_M.x + v.y * gradA_M.y + v.z * gradA_M.z) / (v_len * g_len);
        dot = fmaxf(-1.0f, fminf(1.0f, dot));
        alignment = dot * dot; // cos^2(angle)
    }
    float tension_penalty = expf(-lambda1 * fabsf(dtension));
    float hf_penalty = 1.0f - tanhf(lambda2 * A_H);
    return alignment * tension_penalty * hf_penalty;
}

__host__ __device__ inline void compute_metric_tensor(
    float tau, float dtension, float3 v, float4 audio, float3 accel, float phase,
    float4& diag_gamma, float4& offdiag_phase,
    const SensoryPhaseConfig& cfg)
{
    float A_L = audio.x;
    float A_M = audio.y;
    float A_H = audio.z;

    // 1. Isotropic expansion factor g_iso
    float g_iso = 1.0f + cfg.kappa_1 * tau * (1.0f + A_L);

    // 2. Velocity outer product (v x v)
    float v_xx = v.x * v.x;
    float v_yy = v.y * v.y;
    float v_zz = v.z * v.z;
    float v_xy = v.x * v.y;
    float v_xz = v.x * v.z;
    float v_yz = v.y * v.z;

    // 3. Audio directional bases (e_x, e_y, e_z) distortion
    float M_xx = g_iso + cfg.alpha * v_xx + cfg.beta * A_L;
    float M_yy = g_iso + cfg.alpha * v_yy + cfg.beta * A_M;
    float M_zz = g_iso + cfg.alpha * v_zz + cfg.beta * A_H;

    float M_xy = cfg.alpha * v_xy;
    float M_xz = cfg.alpha * v_xz;
    float M_yz = cfg.alpha * v_yz;

    // 4. Local time dilation factor gamma_i = 1.0 + gamma_max * (1.0 - Phi_i) * tanh(||a_i|| + lambda_3 * |dtau/dt|)
    float accel_norm = sqrtf(accel.x * accel.x + accel.y * accel.y + accel.z * accel.z);
    float gamma = 1.0f + cfg.gamma_max * (1.0f - phase) * tanhf(accel_norm + cfg.lambda_3 * fabsf(dtension));

    diag_gamma.x = M_xx;
    diag_gamma.y = M_yy;
    diag_gamma.z = M_zz;
    diag_gamma.w = gamma;

    offdiag_phase.x = M_xy;
    offdiag_phase.y = M_xz;
    offdiag_phase.z = M_yz;
    offdiag_phase.w = phase;
}

__host__ __device__ inline float3 compute_eigenvalues_3x3(
    float m00, float m01, float m02,
    float m11, float m12, float m22)
{
    float p1 = m01 * m01 + m02 * m02 + m12 * m12;
    if (p1 == 0.0f) {
        return float3{m00, m11, m22};
    }
    float q = (m00 + m11 + m22) / 3.0f;
    float p2 = (m00 - q) * (m00 - q) + (m11 - q) * (m11 - q) + (m22 - q) * (m22 - q) + 2.0f * p1;
    float p = sqrtf(p2 / 6.0f);

    float b00 = (m00 - q) / p, b11 = (m11 - q) / p, b22 = (m22 - q) / p;
    float b01 = m01 / p, b02 = m02 / p, b12 = m12 / p;
    float detB = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02);
    float r = detB / 2.0f;
    r = fmaxf(-1.0f, fminf(1.0f, r));
    float phi = acosf(r) / 3.0f;

    const float M_PI_F = 3.14159265358979323846f;
    float eig1 = q + 2.0f * p * cosf(phi);
    float eig3 = q + 2.0f * p * cosf(phi + (2.0f * M_PI_F / 3.0f));
    float eig2 = 3.0f * q - eig1 - eig3;
    return float3{eig1, eig2, eig3};
}

// Host C++ API Pipeline
class SensoryPhaseCorePipeline {
public:
    SensoryPhaseCorePipeline(const SensoryPhaseConfig& cfg = SensoryPhaseConfig());

    void process_frame_cpu(
        const std::vector<SensoryStreamInput>& inputs,
        std::vector<SpacetimeNodeBoundary>& nodes,
        std::vector<TensorFieldDiagnostics>& diagnostics);

#if defined(__CUDACC__) || defined(WITH_CUDA)
    void process_frame_cuda(
        const SensoryStreamInput* d_inputs,
        SpacetimeNodeBoundary* d_nodes,
        TensorFieldDiagnostics* d_diagnostics,
        int node_count);
#endif

    const SensoryPhaseConfig& get_config() const { return config_; }

private:
    SensoryPhaseConfig config_;
};

} // namespace Elysia

#endif // SENSORY_PHASE_CORE_HPP
