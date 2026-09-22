#include "sensory_phase_core.hpp"

#if defined(__CUDACC__) || defined(WITH_CUDA)
#include <cuda_runtime.h>

namespace Elysia {

__global__ void SensoryToMetricFieldKernel(
    const SensoryStreamInput* __restrict__ inputs,
    SpacetimeNodeBoundary* __restrict__ nodes,
    TensorFieldDiagnostics* __restrict__ diagnostics,
    SensoryPhaseConfig cfg,
    int nodeCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nodeCount) return;

    SensoryStreamInput in = inputs[idx];

    // 1. Phase coherence Φ calculation
    float3 v = make_float3(in.velocity_dtension.x, in.velocity_dtension.y, in.velocity_dtension.z);
    float dtension = in.velocity_dtension.w;
    float3 gradA_M = make_float3(in.acceleration_grad.x, in.acceleration_grad.y, in.acceleration_grad.z);
    float A_H = in.audio_spectrum.z;

    float phase = calculate_phase_coherence(v, gradA_M, dtension, A_H, cfg.lambda_1, cfg.lambda_2);

    // 2. Metric tensor M and time dilation gamma calculation
    float3 accel = make_float3(in.acceleration_grad.x, in.acceleration_grad.y, in.acceleration_grad.z);
    float tau = in.position_tension.w;

    float4 diag_gamma;
    float4 offdiag_phase;
    compute_metric_tensor(tau, dtension, v, in.audio_spectrum, accel, phase, diag_gamma, offdiag_phase, cfg);

    // 3. Write to 16-Byte aligned VRAM
    nodes[idx].metric_diag = diag_gamma;
    nodes[idx].metric_offdiag = offdiag_phase;
    nodes[idx].csr_topology = make_uint4(idx, 0, 0, 0);

    // 4. Evaluate Hessian & GDI
    float H00 = diag_gamma.x - 1.0f;
    float H11 = diag_gamma.y - 1.0f;
    float H22 = diag_gamma.z - 1.0f;
    float H01 = offdiag_phase.x;
    float H02 = offdiag_phase.y;
    float H12 = offdiag_phase.z;

    float3 eig = compute_eigenvalues_3x3(H00, H01, H02, H11, H12, H22);
    float hessian_det = eig.x * eig.y * eig.z;
    float gdi = -1.0f * (eig.x * eig.z);

    uint32_t state = 0;
    if (fabsf(dtension) > cfg.tau_shear) {
        state = 3; // Shear
    } else if (phase >= cfg.phi_solid) {
        state = 2; // Solid
    } else if (phase >= cfg.phi_liquid || (eig.x > 0.0f && eig.z < 0.0f)) {
        state = 1; // Liquid
    }

    uint32_t attractor_id = (gdi < 0.0f) ? 1 : 2;

    diagnostics[idx].position_gdi = make_float4(in.position_tension.x, in.position_tension.y, in.position_tension.z, gdi);
    diagnostics[idx].saddle_hessian = make_float4(hessian_det, eig.x, eig.z, phase);
    diagnostics[idx].classification_st = make_uint4(state, attractor_id, 0, 0);
}

void SensoryPhaseCorePipeline::process_frame_cuda(
    const SensoryStreamInput* d_inputs,
    SpacetimeNodeBoundary* d_nodes,
    TensorFieldDiagnostics* d_diagnostics,
    int node_count)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (node_count + threadsPerBlock - 1) / threadsPerBlock;

    SensoryToMetricFieldKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_inputs, d_nodes, d_diagnostics, config_, node_count);
    cudaDeviceSynchronize();
}

} // namespace Elysia
#endif
