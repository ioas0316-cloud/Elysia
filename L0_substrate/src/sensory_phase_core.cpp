#include "sensory_phase_core.hpp"
#include <iostream>

namespace Elysia {

SensoryPhaseCorePipeline::SensoryPhaseCorePipeline(const SensoryPhaseConfig& cfg)
    : config_(cfg) {}

void SensoryPhaseCorePipeline::process_frame_cpu(
    const std::vector<SensoryStreamInput>& inputs,
    std::vector<SpacetimeNodeBoundary>& nodes,
    std::vector<TensorFieldDiagnostics>& diagnostics)
{
    size_t count = inputs.size();
    nodes.resize(count);
    diagnostics.resize(count);

    for (size_t i = 0; i < count; ++i) {
        const auto& in = inputs[i];

        float3 v{in.velocity_dtension.x, in.velocity_dtension.y, in.velocity_dtension.z};
        float dtension = in.velocity_dtension.w;
        float3 gradA_M{in.acceleration_grad.x, in.acceleration_grad.y, in.acceleration_grad.z};
        float A_H = in.audio_spectrum.z;

        // 1. Calculate phase coherence
        float phase = calculate_phase_coherence(
            v, gradA_M, dtension, A_H,
            config_.lambda_1, config_.lambda_2);

        // 2. Calculate metric tensor & time dilation
        float3 accel{in.acceleration_grad.x, in.acceleration_grad.y, in.acceleration_grad.z};
        float tau = in.position_tension.w;

        float4 diag_gamma, offdiag_phase;
        compute_metric_tensor(tau, dtension, v, in.audio_spectrum, accel, phase, diag_gamma, offdiag_phase, config_);

        nodes[i].metric_diag = diag_gamma;
        nodes[i].metric_offdiag = offdiag_phase;
        nodes[i].csr_topology = uint4{static_cast<uint32_t>(i), 0, 0, 0};

        // 3. Compute Hessian & Geodesic Divergence Indicator (GDI)
        float H00 = diag_gamma.x - 1.0f;
        float H11 = diag_gamma.y - 1.0f;
        float H22 = diag_gamma.z - 1.0f;
        float H01 = offdiag_phase.x;
        float H02 = offdiag_phase.y;
        float H12 = offdiag_phase.z;

        float3 eig = compute_eigenvalues_3x3(H00, H01, H02, H11, H12, H22);
        float hessian_det = eig.x * eig.y * eig.z;

        // Local Sectional Curvature GDI ≈ - (λ_max * λ_min)
        float gdi = -1.0f * (eig.x * eig.z);

        // 4. Autonomous phase transition state determination
        uint32_t phase_state = 0; // 0: Gas
        if (fabsf(dtension) > config_.tau_shear) {
            phase_state = 3; // Shear
        } else if (phase >= config_.phi_solid) {
            phase_state = 2; // Solid
        } else if (phase >= config_.phi_liquid || (eig.x > 0.0f && eig.z < 0.0f)) {
            phase_state = 1; // Liquid
        }

        uint32_t attractor_id = (gdi < 0.0f) ? 1 : 2;

        diagnostics[i].position_gdi = float4{in.position_tension.x, in.position_tension.y, in.position_tension.z, gdi};
        diagnostics[i].saddle_hessian = float4{hessian_det, eig.x, eig.z, phase};
        diagnostics[i].classification_st = uint4{phase_state, attractor_id, 0, 0};
    }
}

} // namespace Elysia
