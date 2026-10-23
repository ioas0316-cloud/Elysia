#include "elysia_foc_kernel.h"
#include <cmath>
#include <algorithm>
#include <iostream>

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <math_constants.h>
#include <device_launch_parameters.h>

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void elysia_foc_fused_rotor_kernel(
    const float* __restrict__ input_abc,
    const float* __restrict__ angles,
    float* __restrict__ output_dq0,
    const float gamma_d,
    const int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float a = input_abc[idx * 3 + 0];
    float b = input_abc[idx * 3 + 1];
    float c = input_abc[idx * 3 + 2];
    float theta = angles[idx];

    constexpr float SQRT3_INV_2 = 0.86602540378f;
    constexpr float ONE_THIRD   = 0.33333333333f;

    float v_alpha = a - 0.5f * (b + c);
    float v_beta  = SQRT3_INV_2 * (b - c);
    float v_zero  = ONE_THIRD * (a + b + c);

    float sin_t, cos_t;
    __sincosf(theta, &sin_t, &cos_t);

    float raw_v_d = fmaf(v_alpha, cos_t, v_beta * sin_t);
    float v_d     = raw_v_d * gamma_d; // Flux weakening context scaling
    float v_q     = fmaf(-v_alpha, sin_t, v_beta * cos_t); // Dynamic momentum preservation

    output_dq0[idx * 3 + 0] = v_d;
    output_dq0[idx * 3 + 1] = v_q;
    output_dq0[idx * 3 + 2] = v_zero;
}

__device__ inline Multivector3D apply_morphism_rotor_device(const Multivector3D& v, const Multivector3D& R) {
    Multivector3D result;
    float R_s = R.s;
    float R_b12 = R.b12;

    result.v1 = fmaf(v.v1, R_s * R_s - R_b12 * R_b12, -2.0f * R_s * R_b12 * v.v2);
    result.v2 = fmaf(v.v2, R_s * R_s - R_b12 * R_b12,  2.0f * R_s * R_b12 * v.v1);
    result.v3 = v.v3;

    result.s = v.s;
    result.b12 = v.b12; result.b23 = v.b23; result.b31 = v.b31;
    result.p = v.p;
    return result;
}

__global__ void elysia_yoneda_embedding_kernel(
    const Multivector3D* __restrict__ concept_A,
    const Multivector3D* __restrict__ basis_morphisms,
    float* __restrict__ yoneda_spectrum,
    int num_morphisms)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_morphisms) return;

    Multivector3D A = *concept_A;
    Multivector3D R_k = basis_morphisms[idx];

    Multivector3D A_transformed = apply_morphism_rotor_device(A, R_k);
    float coherence = fmaf(A_transformed.v1, A.v1, fmaf(A_transformed.v2, A.v2, A_transformed.v3 * A.v3));
    yoneda_spectrum[idx] = coherence;
}

__global__ void elysia_renormalization_group_3x3x3_kernel(
    const Multivector3D* __restrict__ level_L_nodes,
    Multivector3D* __restrict__ level_L_plus_1_nodes,
    float* __restrict__ coherence_factors,
    int total_blocks)
{
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) return;

    int base_offset = block_idx * 27;

    float sum_v1 = 0.0f, sum_v2 = 0.0f, sum_v3 = 0.0f;
    float norm_sum_v1 = 0.0f, norm_sum_v2 = 0.0f, norm_sum_v3 = 0.0f;

    #pragma unroll
    for (int i = 0; i < 27; ++i) {
        Multivector3D node = level_L_nodes[base_offset + i];
        sum_v1 += node.v1;
        sum_v2 += node.v2;
        sum_v3 += node.v3;

        float inv_len = rsqrtf(fmaf(node.v1, node.v1, fmaf(node.v2, node.v2, node.v3 * node.v3)) + 1e-8f);
        norm_sum_v1 += node.v1 * inv_len;
        norm_sum_v2 += node.v2 * inv_len;
        norm_sum_v3 += node.v3 * inv_len;
    }

    float coherence_len = sqrtf(norm_sum_v1 * norm_sum_v1 + norm_sum_v2 * norm_sum_v2 + norm_sum_v3 * norm_sum_v3);
    float omega = coherence_len * (1.0f / 27.0f);

    float scale_factor = (omega > 0.2f) ? (omega / 27.0f) : 0.0f;

    Multivector3D effective_field = {0.0f};
    effective_field.v1 = sum_v1 * scale_factor;
    effective_field.v2 = sum_v2 * scale_factor;
    effective_field.v3 = sum_v3 * scale_factor;

    level_L_plus_1_nodes[block_idx] = effective_field;
    coherence_factors[block_idx] = omega;
}

__global__ void elysia_hippocampal_stdp_rotor_kernel(
    Rotor3D* __restrict__ rotors,
    const float* __restrict__ t_pre,
    const float* __restrict__ t_post,
    float A_plus, float A_minus,
    float tau_plus, float tau_minus,
    int num_synapses)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_synapses) return;

    float delta_t = t_post[idx] - t_pre[idx];
    float delta_theta = 0.0f;

    if (delta_t > 0.0f) {
        delta_theta = A_plus * __expf(-delta_t / tau_plus);
    } else if (delta_t < 0.0f) {
        delta_theta = -A_minus * __expf(delta_t / tau_minus);
    }

    float half_angle = delta_theta * 0.5f;
    float sin_h = __sinf(half_angle);
    float cos_h = __cosf(half_angle);

    Rotor3D current_R = rotors[idx];
    Rotor3D updated_R;
    updated_R.s   = fmaf(current_R.s, cos_h, -current_R.b12 * sin_h);
    updated_R.b12 = fmaf(current_R.s, sin_h,  current_R.b12 * cos_h);
    updated_R.b23 = fmaf(current_R.b23, cos_h, -current_R.b31 * sin_h);
    updated_R.b31 = fmaf(current_R.b31, cos_h,  current_R.b23 * sin_h);

    float norm = rsqrtf(fmaf(updated_R.s, updated_R.s,
                       fmaf(updated_R.b12, updated_R.b12,
                       fmaf(updated_R.b23, updated_R.b23, updated_R.b31 * updated_R.b31))));

    rotors[idx] = {updated_R.s * norm, updated_R.b12 * norm, updated_R.b23 * norm, updated_R.b31 * norm};
}

__global__ void elysia_gaba_rg_gating_kernel(
    const Multivector3D* __restrict__ micro_nodes,
    Multivector3D* __restrict__ macro_nodes,
    float omega_threshold,
    float beta,
    int total_blocks)
{
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) return;

    int base_offset = block_idx * 27;

    float sum_v1 = 0.0f, sum_v2 = 0.0f, sum_v3 = 0.0f;
    float norm_sum_v1 = 0.0f, norm_sum_v2 = 0.0f, norm_sum_v3 = 0.0f;

    #pragma unroll
    for (int i = 0; i < 27; ++i) {
        Multivector3D node = micro_nodes[base_offset + i];
        sum_v1 += node.v1; sum_v2 += node.v2; sum_v3 += node.v3;

        float inv_len = rsqrtf(fmaf(node.v1, node.v1, fmaf(node.v2, node.v2, node.v3 * node.v3)) + 1e-8f);
        norm_sum_v1 += node.v1 * inv_len;
        norm_sum_v2 += node.v2 * inv_len;
        norm_sum_v3 += node.v3 * inv_len;
    }

    float coherence_len = sqrtf(norm_sum_v1 * norm_sum_v1 + norm_sum_v2 * norm_sum_v2 + norm_sum_v3 * norm_sum_v3);
    float omega = coherence_len * (1.0f / 27.0f);

    float gaba_gate = 1.0f / (1.0f + __expf(-beta * (omega - omega_threshold)));

    Multivector3D macro_field = {0.0f};
    float scale = (gaba_gate / 27.0f);

    macro_field.v1 = sum_v1 * scale;
    macro_field.v2 = sum_v2 * scale;
    macro_field.v3 = sum_v3 * scale;

    macro_nodes[block_idx] = macro_field;
}

__global__ void elysia_phase_crystallization_kernel(
    const Rotor3D* __restrict__ dynamic_rotors,
    Rotor3D* __restrict__ static_memory_rotors,
    unsigned char* __restrict__ is_crystallized,
    float omega_th,
    float eta_crit,
    int num_rotors)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rotors) return;

    Rotor3D r_dyn = dynamic_rotors[idx];
    float r_norm_sq = fmaf(r_dyn.s, r_dyn.s, fmaf(r_dyn.b12, r_dyn.b12, fmaf(r_dyn.b23, r_dyn.b23, r_dyn.b31 * r_dyn.b31)));
    float eta = sqrtf(r_norm_sq);

    bool transition_condition = (eta >= eta_crit) && (r_dyn.s >= omega_th);

    if (transition_condition) {
        float inv_eta = 1.0f / (eta + 1e-8f);
        Rotor3D frozen_rotor = {
            r_dyn.s * inv_eta,
            r_dyn.b12 * inv_eta,
            r_dyn.b23 * inv_eta,
            r_dyn.b31 * inv_eta
        };

        static_memory_rotors[idx] = frozen_rotor;
        is_crystallized[idx] = 1;
    } else {
        is_crystallized[idx] = 0;
    }
}

__global__ void elysia_open_system_receptor_kernel(
    const Multivector3D* __restrict__ external_gas_waves,
    Multivector3D* __restrict__ dynamic_rotor_fluid,
    float receptor_coupling_gamma,
    float dt,
    int num_receptors)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_receptors) return;

    Multivector3D psi_ext = external_gas_waves[idx];
    Multivector3D r_fluid = dynamic_rotor_fluid[idx];

    float delta_b12 = receptor_coupling_gamma * (psi_ext.b12 - r_fluid.b12) * dt;
    float delta_b23 = receptor_coupling_gamma * (psi_ext.b23 - r_fluid.b23) * dt;
    float delta_b31 = receptor_coupling_gamma * (psi_ext.b31 - r_fluid.b31) * dt;

    r_fluid.b12 += delta_b12;
    r_fluid.b23 += delta_b23;
    r_fluid.b31 += delta_b31;

    float norm = rsqrtf(fmaf(r_fluid.s, r_fluid.s,
                       fmaf(r_fluid.b12, r_fluid.b12,
                       fmaf(r_fluid.b23, r_fluid.b23, r_fluid.b31 * r_fluid.b31))) + 1e-8f);

    r_fluid.s   *= norm;
    r_fluid.b12 *= norm;
    r_fluid.b23 *= norm;
    r_fluid.b31 *= norm;

    dynamic_rotor_fluid[idx] = r_fluid;
}

__global__ void k_pyramidal_mismatch_rpe_fusion(
    const Rotor3D* __restrict__ g_psi_top,
    const Rotor3D* __restrict__ g_psi_bot,
    Rotor3D* __restrict__ g_rotors,
    float lambda_sensitivity,
    float noise_seed,
    int num_neurons)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_neurons) return;

    Rotor3D top = g_psi_top[tid];
    Rotor3D bot = g_psi_bot[tid];
    Rotor3D bot_dag = { bot.s, -bot.b12, -bot.b23, -bot.b31 };

    float b12 = top.s * bot_dag.b12 + top.b12 * bot_dag.s + (top.b23 * bot_dag.b31 - top.b31 * bot_dag.b23);
    float b23 = top.s * bot_dag.b23 + top.b23 * bot_dag.s + (top.b31 * bot_dag.b12 - top.b12 * bot_dag.b31);
    float b31 = top.s * bot_dag.b31 + top.b31 * bot_dag.s + (top.b12 * bot_dag.b23 - top.b23 * bot_dag.b12);

    float local_error = sqrtf(b12 * b12 + b23 * b23 + b31 * b31);

    float warp_error_sum = local_error;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        warp_error_sum += __shfl_down_sync(0xffffffff, warp_error_sum, offset);
    }
    float collective_mismatch = __shfl_sync(0xffffffff, warp_error_sum, 0);

    float mu = expf(-lambda_sensitivity * collective_mismatch);
    float noise_scale = sqrtf(fmaxf(0.0f, 1.0f - mu * mu));

    float rng = fmodf(sinf(tid * 12.9898f + noise_seed) * 43758.5453f, 6.28318f);
    Rotor3D r_old = g_rotors[tid];

    g_rotors[tid].s   = mu * r_old.s   + noise_scale * cosf(rng);
    g_rotors[tid].b12 = mu * r_old.b12 + noise_scale * sinf(rng) * 0.5773f;
    g_rotors[tid].b23 = mu * r_old.b23 + noise_scale * sinf(rng) * 0.5773f;
    g_rotors[tid].b31 = mu * r_old.b31 + noise_scale * sinf(rng) * 0.5773f;
}

__global__ void k_rotor_crystallization_fusion(
    const Rotor3D* __restrict__ g_psi_top,
    const Rotor3D* __restrict__ g_psi_bot,
    Rotor3D* __restrict__ g_rotors,
    float ach_level,
    float freeze_rate,
    float coherence_threshold,
    int num_neurons)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_neurons) return;

    Rotor3D top = g_psi_top[tid];
    Rotor3D bot = g_psi_bot[tid];
    Rotor3D r_curr = g_rotors[tid];

    Rotor3D bot_dag = { bot.s, -bot.b12, -bot.b23, -bot.b31 };
    float coherence = top.s * bot_dag.s - (top.b12 * bot_dag.b12 + top.b23 * bot_dag.b23 + top.b31 * bot_dag.b31);

    if (coherence >= coherence_threshold) {
        float cooling_factor = (1.0f - ach_level) * freeze_rate * coherence;

        float s_new   = r_curr.s   + cooling_factor * top.s;
        float b12_new = r_curr.b12 + cooling_factor * top.b12;
        float b23_new = r_curr.b23 + cooling_factor * top.b23;
        float b31_new = r_curr.b31 + cooling_factor * top.b31;

        float norm_sq = s_new * s_new + b12_new * b12_new + b23_new * b23_new + b31_new * b31_new;
        float inv_norm = rsqrtf(fmaxf(norm_sq, 1e-8f));

        g_rotors[tid].s   = s_new   * inv_norm;
        g_rotors[tid].b12 = b12_new * inv_norm;
        g_rotors[tid].b23 = b23_new * inv_norm;
        g_rotors[tid].b31 = b31_new * inv_norm;
    }
}

#endif // __CUDACC__

// ============================================================================
// C++ / Host Launch Functions with CPU Fallback Support
// ============================================================================

extern "C" {

void launch_elysia_foc_kernel(
    const float* d_abc,
    const float* d_angles,
    float* d_dq0,
    float gamma_d,
    int total_elements,
    cudaStream_t stream)
{
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    elysia_foc_fused_rotor_kernel<<<grid_size, block_size, 0, stream>>>(
        d_abc, d_angles, d_dq0, gamma_d, total_elements
    );
#else
    // CPU Host Fallback Loop
    constexpr float SQRT3_INV_2 = 0.86602540378f;
    constexpr float ONE_THIRD   = 0.33333333333f;

    for (int idx = 0; idx < total_elements; ++idx) {
        float a = d_abc[idx * 3 + 0];
        float b = d_abc[idx * 3 + 1];
        float c = d_abc[idx * 3 + 2];
        float theta = d_angles[idx];

        float v_alpha = a - 0.5f * (b + c);
        float v_beta  = SQRT3_INV_2 * (b - c);
        float v_zero  = ONE_THIRD * (a + b + c);

        float sin_t = std::sin(theta);
        float cos_t = std::cos(theta);

        float raw_v_d = v_alpha * cos_t + v_beta * sin_t;
        float v_d     = raw_v_d * gamma_d;
        float v_q     = -v_alpha * sin_t + v_beta * cos_t;

        d_dq0[idx * 3 + 0] = v_d;
        d_dq0[idx * 3 + 1] = v_q;
        d_dq0[idx * 3 + 2] = v_zero;
    }
#endif
}

void launch_elysia_yoneda_embedding_kernel(
    const Multivector3D* d_concept_A,
    const Multivector3D* d_basis_morphisms,
    float* d_yoneda_spectrum,
    int num_morphisms,
    cudaStream_t stream)
{
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
    int block_size = 256;
    int grid_size = (num_morphisms + block_size - 1) / block_size;
    elysia_yoneda_embedding_kernel<<<grid_size, block_size, 0, stream>>>(
        d_concept_A, d_basis_morphisms, d_yoneda_spectrum, num_morphisms
    );
#else
    Multivector3D A = *d_concept_A;
    for (int idx = 0; idx < num_morphisms; ++idx) {
        Multivector3D R = d_basis_morphisms[idx];
        float R_s = R.s;
        float R_b12 = R.b12;

        Multivector3D A_transformed;
        A_transformed.v1 = A.v1 * (R_s * R_s - R_b12 * R_b12) - 2.0f * R_s * R_b12 * A.v2;
        A_transformed.v2 = A.v2 * (R_s * R_s - R_b12 * R_b12) + 2.0f * R_s * R_b12 * A.v1;
        A_transformed.v3 = A.v3;

        float coherence = A_transformed.v1 * A.v1 + A_transformed.v2 * A.v2 + A_transformed.v3 * A.v3;
        d_yoneda_spectrum[idx] = coherence;
    }
#endif
}

void launch_elysia_rg_3x3x3_kernel(
    const Multivector3D* d_level_L_nodes,
    Multivector3D* d_level_L_plus_1_nodes,
    float* d_coherence_factors,
    int total_blocks,
    cudaStream_t stream)
{
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
    int block_size = 256;
    int grid_size = (total_blocks + block_size - 1) / block_size;
    elysia_renormalization_group_3x3x3_kernel<<<grid_size, block_size, 0, stream>>>(
        d_level_L_nodes, d_level_L_plus_1_nodes, d_coherence_factors, total_blocks
    );
#else
    for (int block_idx = 0; block_idx < total_blocks; ++block_idx) {
        int base_offset = block_idx * 27;

        float sum_v1 = 0.0f, sum_v2 = 0.0f, sum_v3 = 0.0f;
        float norm_sum_v1 = 0.0f, norm_sum_v2 = 0.0f, norm_sum_v3 = 0.0f;

        for (int i = 0; i < 27; ++i) {
            Multivector3D node = d_level_L_nodes[base_offset + i];
            sum_v1 += node.v1; sum_v2 += node.v2; sum_v3 += node.v3;

            float len = std::sqrt(node.v1 * node.v1 + node.v2 * node.v2 + node.v3 * node.v3 + 1e-8f);
            float inv_len = 1.0f / len;
            norm_sum_v1 += node.v1 * inv_len;
            norm_sum_v2 += node.v2 * inv_len;
            norm_sum_v3 += node.v3 * inv_len;
        }

        float coherence_len = std::sqrt(norm_sum_v1 * norm_sum_v1 + norm_sum_v2 * norm_sum_v2 + norm_sum_v3 * norm_sum_v3);
        float omega = coherence_len * (1.0f / 27.0f);
        float scale_factor = (omega > 0.2f) ? (omega / 27.0f) : 0.0f;

        Multivector3D effective_field = {0.0f};
        effective_field.v1 = sum_v1 * scale_factor;
        effective_field.v2 = sum_v2 * scale_factor;
        effective_field.v3 = sum_v3 * scale_factor;

        d_level_L_plus_1_nodes[block_idx] = effective_field;
        d_coherence_factors[block_idx] = omega;
    }
#endif
}

void launch_elysia_stdp_rotor_kernel(
    Rotor3D* d_rotors,
    const float* d_t_pre,
    const float* d_t_post,
    float A_plus, float A_minus,
    float tau_plus, float tau_minus,
    int num_synapses,
    cudaStream_t stream)
{
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
    int block_size = 256;
    int grid_size = (num_synapses + block_size - 1) / block_size;
    elysia_hippocampal_stdp_rotor_kernel<<<grid_size, block_size, 0, stream>>>(
        d_rotors, d_t_pre, d_t_post, A_plus, A_minus, tau_plus, tau_minus, num_synapses
    );
#else
    for (int idx = 0; idx < num_synapses; ++idx) {
        float delta_t = d_t_post[idx] - d_t_pre[idx];
        float delta_theta = 0.0f;

        if (delta_t > 0.0f) {
            delta_theta = A_plus * std::exp(-delta_t / tau_plus);
        } else if (delta_t < 0.0f) {
            delta_theta = -A_minus * std::exp(delta_t / tau_minus);
        }

        float half_angle = delta_theta * 0.5f;
        float sin_h = std::sin(half_angle);
        float cos_h = std::cos(half_angle);

        Rotor3D current_R = d_rotors[idx];
        Rotor3D updated_R;
        updated_R.s   = current_R.s * cos_h - current_R.b12 * sin_h;
        updated_R.b12 = current_R.s * sin_h + current_R.b12 * cos_h;
        updated_R.b23 = current_R.b23 * cos_h - current_R.b31 * sin_h;
        updated_R.b31 = current_R.b31 * cos_h + current_R.b23 * sin_h;

        float norm_sq = updated_R.s * updated_R.s + updated_R.b12 * updated_R.b12 + updated_R.b23 * updated_R.b23 + updated_R.b31 * updated_R.b31;
        float norm = 1.0f / std::sqrt(std::max(norm_sq, 1e-8f));

        d_rotors[idx] = {updated_R.s * norm, updated_R.b12 * norm, updated_R.b23 * norm, updated_R.b31 * norm};
    }
#endif
}

void launch_elysia_gaba_rg_gating_kernel(
    const Multivector3D* d_micro_nodes,
    Multivector3D* d_macro_nodes,
    float omega_threshold,
    float beta,
    int total_blocks,
    cudaStream_t stream)
{
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
    int block_size = 256;
    int grid_size = (total_blocks + block_size - 1) / block_size;
    elysia_gaba_rg_gating_kernel<<<grid_size, block_size, 0, stream>>>(
        d_micro_nodes, d_macro_nodes, omega_threshold, beta, total_blocks
    );
#else
    for (int block_idx = 0; block_idx < total_blocks; ++block_idx) {
        int base_offset = block_idx * 27;

        float sum_v1 = 0.0f, sum_v2 = 0.0f, sum_v3 = 0.0f;
        float norm_sum_v1 = 0.0f, norm_sum_v2 = 0.0f, norm_sum_v3 = 0.0f;

        for (int i = 0; i < 27; ++i) {
            Multivector3D node = d_micro_nodes[base_offset + i];
            sum_v1 += node.v1; sum_v2 += node.v2; sum_v3 += node.v3;

            float inv_len = 1.0f / std::sqrt(node.v1 * node.v1 + node.v2 * node.v2 + node.v3 * node.v3 + 1e-8f);
            norm_sum_v1 += node.v1 * inv_len;
            norm_sum_v2 += node.v2 * inv_len;
            norm_sum_v3 += node.v3 * inv_len;
        }

        float coherence_len = std::sqrt(norm_sum_v1 * norm_sum_v1 + norm_sum_v2 * norm_sum_v2 + norm_sum_v3 * norm_sum_v3);
        float omega = coherence_len * (1.0f / 27.0f);
        float gaba_gate = 1.0f / (1.0f + std::exp(-beta * (omega - omega_threshold)));

        Multivector3D macro_field = {0.0f};
        float scale = (gaba_gate / 27.0f);

        macro_field.v1 = sum_v1 * scale;
        macro_field.v2 = sum_v2 * scale;
        macro_field.v3 = sum_v3 * scale;

        d_macro_nodes[block_idx] = macro_field;
    }
#endif
}

void launch_elysia_phase_crystallization_kernel(
    const Rotor3D* d_dynamic_rotors,
    Rotor3D* d_static_memory_rotors,
    unsigned char* d_is_crystallized,
    float omega_th,
    float eta_crit,
    int num_rotors,
    cudaStream_t stream)
{
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
    int block_size = 256;
    int grid_size = (num_rotors + block_size - 1) / block_size;
    elysia_phase_crystallization_kernel<<<grid_size, block_size, 0, stream>>>(
        d_dynamic_rotors, d_static_memory_rotors, d_is_crystallized, omega_th, eta_crit, num_rotors
    );
#else
    for (int idx = 0; idx < num_rotors; ++idx) {
        Rotor3D r_dyn = d_dynamic_rotors[idx];
        float r_norm_sq = r_dyn.s * r_dyn.s + r_dyn.b12 * r_dyn.b12 + r_dyn.b23 * r_dyn.b23 + r_dyn.b31 * r_dyn.b31;
        float eta = std::sqrt(r_norm_sq);

        bool transition_condition = (eta >= eta_crit) && (r_dyn.s >= omega_th);

        if (transition_condition) {
            float inv_eta = 1.0f / (eta + 1e-8f);
            Rotor3D frozen_rotor = {
                r_dyn.s * inv_eta,
                r_dyn.b12 * inv_eta,
                r_dyn.b23 * inv_eta,
                r_dyn.b31 * inv_eta
            };

            d_static_memory_rotors[idx] = frozen_rotor;
            d_is_crystallized[idx] = 1;
        } else {
            d_is_crystallized[idx] = 0;
        }
    }
#endif
}

void launch_elysia_open_system_receptor_kernel(
    const Multivector3D* d_external_gas_waves,
    Multivector3D* d_dynamic_rotor_fluid,
    float receptor_coupling_gamma,
    float dt,
    int num_receptors,
    cudaStream_t stream)
{
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
    int block_size = 256;
    int grid_size = (num_receptors + block_size - 1) / block_size;
    elysia_open_system_receptor_kernel<<<grid_size, block_size, 0, stream>>>(
        d_external_gas_waves, d_dynamic_rotor_fluid, receptor_coupling_gamma, dt, num_receptors
    );
#else
    for (int idx = 0; idx < num_receptors; ++idx) {
        Multivector3D psi_ext = d_external_gas_waves[idx];
        Multivector3D r_fluid = d_dynamic_rotor_fluid[idx];

        float delta_b12 = receptor_coupling_gamma * (psi_ext.b12 - r_fluid.b12) * dt;
        float delta_b23 = receptor_coupling_gamma * (psi_ext.b23 - r_fluid.b23) * dt;
        float delta_b31 = receptor_coupling_gamma * (psi_ext.b31 - r_fluid.b31) * dt;

        r_fluid.b12 += delta_b12;
        r_fluid.b23 += delta_b23;
        r_fluid.b31 += delta_b31;

        float norm_sq = r_fluid.s * r_fluid.s + r_fluid.b12 * r_fluid.b12 + r_fluid.b23 * r_fluid.b23 + r_fluid.b31 * r_fluid.b31;
        float norm = 1.0f / std::sqrt(std::max(norm_sq, 1e-8f));

        r_fluid.s   *= norm;
        r_fluid.b12 *= norm;
        r_fluid.b23 *= norm;
        r_fluid.b31 *= norm;

        d_dynamic_rotor_fluid[idx] = r_fluid;
    }
#endif
}

void launch_elysia_pyramidal_mismatch_rpe_fusion_kernel(
    const Rotor3D* d_psi_top,
    const Rotor3D* d_psi_bot,
    Rotor3D* d_rotors,
    float lambda_sensitivity,
    float noise_seed,
    int num_neurons,
    cudaStream_t stream)
{
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
    int block_size = 256;
    int grid_size = (num_neurons + block_size - 1) / block_size;
    k_pyramidal_mismatch_rpe_fusion<<<grid_size, block_size, 0, stream>>>(
        d_psi_top, d_psi_bot, d_rotors, lambda_sensitivity, noise_seed, num_neurons
    );
#else
    float total_error = 0.0f;
    for (int tid = 0; tid < num_neurons; ++tid) {
        Rotor3D top = d_psi_top[tid];
        Rotor3D bot = d_psi_bot[tid];
        Rotor3D bot_dag = { bot.s, -bot.b12, -bot.b23, -bot.b31 };

        float b12 = top.s * bot_dag.b12 + top.b12 * bot_dag.s + (top.b23 * bot_dag.b31 - top.b31 * bot_dag.b23);
        float b23 = top.s * bot_dag.b23 + top.b23 * bot_dag.s + (top.b31 * bot_dag.b12 - top.b12 * bot_dag.b31);
        float b31 = top.s * bot_dag.b31 + top.b31 * bot_dag.s + (top.b12 * bot_dag.b23 - top.b23 * bot_dag.b12);

        float local_error = std::sqrt(b12 * b12 + b23 * b23 + b31 * b31);
        total_error += local_error;
    }

    float collective_mismatch = (num_neurons > 0) ? (total_error / num_neurons) : 0.0f;
    float mu = std::exp(-lambda_sensitivity * collective_mismatch);
    float noise_scale = std::sqrt(std::max(0.0f, 1.0f - mu * mu));

    for (int tid = 0; tid < num_neurons; ++tid) {
        float rng = std::fmod(std::sin(tid * 12.9898f + noise_seed) * 43758.5453f, 6.28318f);
        Rotor3D r_old = d_rotors[tid];

        d_rotors[tid].s   = mu * r_old.s   + noise_scale * std::cos(rng);
        d_rotors[tid].b12 = mu * r_old.b12 + noise_scale * std::sin(rng) * 0.5773f;
        d_rotors[tid].b23 = mu * r_old.b23 + noise_scale * std::sin(rng) * 0.5773f;
        d_rotors[tid].b31 = mu * r_old.b31 + noise_scale * std::sin(rng) * 0.5773f;
    }
#endif
}

void launch_elysia_rotor_crystallization_fusion_kernel(
    const Rotor3D* d_psi_top,
    const Rotor3D* d_psi_bot,
    Rotor3D* d_rotors,
    float ach_level,
    float freeze_rate,
    float coherence_threshold,
    int num_neurons,
    cudaStream_t stream)
{
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
    int block_size = 256;
    int grid_size = (num_neurons + block_size - 1) / block_size;
    k_rotor_crystallization_fusion<<<grid_size, block_size, 0, stream>>>(
        d_psi_top, d_psi_bot, d_rotors, ach_level, freeze_rate, coherence_threshold, num_neurons
    );
#else
    for (int tid = 0; tid < num_neurons; ++tid) {
        Rotor3D top = d_psi_top[tid];
        Rotor3D bot = d_psi_bot[tid];
        Rotor3D r_curr = d_rotors[tid];

        Rotor3D bot_dag = { bot.s, -bot.b12, -bot.b23, -bot.b31 };
        float coherence = top.s * bot_dag.s - (top.b12 * bot_dag.b12 + top.b23 * bot_dag.b23 + top.b31 * bot_dag.b31);

        if (coherence >= coherence_threshold) {
            float cooling_factor = (1.0f - ach_level) * freeze_rate * coherence;

            float s_new   = r_curr.s   + cooling_factor * top.s;
            float b12_new = r_curr.b12 + cooling_factor * top.b12;
            float b23_new = r_curr.b23 + cooling_factor * top.b23;
            float b31_new = r_curr.b31 + cooling_factor * top.b31;

            float norm_sq = s_new * s_new + b12_new * b12_new + b23_new * b23_new + b31_new * b31_new;
            float inv_norm = 1.0f / std::sqrt(std::max(norm_sq, 1e-8f));

            d_rotors[tid].s   = s_new   * inv_norm;
            d_rotors[tid].b12 = b12_new * inv_norm;
            d_rotors[tid].b23 = b23_new * inv_norm;
            d_rotors[tid].b31 = b31_new * inv_norm;
        }
    }
#endif
}

} // extern "C"
