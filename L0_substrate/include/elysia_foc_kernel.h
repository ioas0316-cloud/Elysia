#ifndef ELYSIA_FOC_KERNEL_H
#define ELYSIA_FOC_KERNEL_H

#include <cstddef>

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#else
#include "elysia/cuda_host_stub.h"
#endif

// ============================================================================
// 1. Data Structures for Clifford Cl(3,0) & FOC Wave Computing
// ============================================================================

// Clifford Cl(3,0) Rotor representation (Scalar + Bivectors)
struct Rotor3D {
    float s;             // Scalar (1)
    float b12, b23, b31; // Bivectors (e12, e23, e31)
};

// Clifford Cl(3,0) Multivector (Scalar + Vector + Bivector + Pseudoscalar)
struct Multivector3D {
    float s;             // Scalar (1)
    float v1, v2, v3;    // Vector (e1, e2, e3)
    float b12, b23, b31; // Bivector (e12, e23, e31)
    float p;             // Pseudoscalar (e123)
};

// Structure of Arrays (SoA) layout for GTX 1060 (Pascal) memory coalescing
struct Rotor3D_SoA {
    float* s;
    float* b12;
    float* b23;
    float* b31;
};

struct Multivector3D_SoA {
    float* v1;
    float* v2;
    float* v3;
};

// Cognitive metrics structure for closed-loop GABA & Flux Weakening control
struct CognitiveLoadMetrics {
    float phase_entropy;      // Phase disorder measure H
    float active_node_ratio;  // Ratio of active non-suppressed nodes
    float vram_used_mb;       // VRAM usage in MB
    float vram_limit_mb;      // VRAM physical/soft limit in MB
};

// Mismatch result evaluation for Pyramidal Neurons
struct MismatchResult {
    float alignment_coherence; // Scalar in-phase alignment
    float error_spike;         // Bivector norm mismatch
};

// ============================================================================
// 2. Kernel Host Launch Interfaces
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Fused Clarke-Park Rotor transformation with Flux Weakening scaling
void launch_elysia_foc_kernel(
    const float* d_abc,
    const float* d_angles,
    float* d_dq0,
    float gamma_d,
    int total_elements,
    cudaStream_t stream = nullptr
);

// Yoneda Lemma spectrum extraction kernel
void launch_elysia_yoneda_embedding_kernel(
    const Multivector3D* d_concept_A,
    const Multivector3D* d_basis_morphisms,
    float* d_yoneda_spectrum,
    int num_morphisms,
    cudaStream_t stream = nullptr
);

// 3x3x3 Fractal Renormalization Group (RG) Coarse-Graining kernel
void launch_elysia_rg_3x3x3_kernel(
    const Multivector3D* d_level_L_nodes,
    Multivector3D* d_level_L_plus_1_nodes,
    float* d_coherence_factors,
    int total_blocks,
    cudaStream_t stream = nullptr
);

// Hippocampal STDP Phase Update kernel
void launch_elysia_stdp_rotor_kernel(
    Rotor3D* d_rotors,
    const float* d_t_pre,
    const float* d_t_post,
    float A_plus, float A_minus,
    float tau_plus, float tau_minus,
    int num_synapses,
    cudaStream_t stream = nullptr
);

// GABAergic Inhibition RG Gating kernel
void launch_elysia_gaba_rg_gating_kernel(
    const Multivector3D* d_micro_nodes,
    Multivector3D* d_macro_nodes,
    float omega_threshold,
    float beta,
    int total_blocks,
    cudaStream_t stream = nullptr
);

// Phase Crystallization (Liquid -> Solid Memory) Transition kernel
void launch_elysia_phase_crystallization_kernel(
    const Rotor3D* d_dynamic_rotors,
    Rotor3D* d_static_memory_rotors,
    unsigned char* d_is_crystallized,
    float omega_th,
    float eta_crit,
    int num_rotors,
    cudaStream_t stream = nullptr
);

// Open-System Receptor Interference (Gas -> Liquid Momentum) kernel
void launch_elysia_open_system_receptor_kernel(
    const Multivector3D* d_external_gas_waves,
    Multivector3D* d_dynamic_rotor_fluid,
    float receptor_coupling_gamma,
    float dt,
    int num_receptors,
    cudaStream_t stream = nullptr
);

// Pyramidal 2-Layer Mismatch and Negative RPE Melting Fusion kernel
void launch_elysia_pyramidal_mismatch_rpe_fusion_kernel(
    const Rotor3D* d_psi_top,
    const Rotor3D* d_psi_bot,
    Rotor3D* d_rotors,
    float lambda_sensitivity,
    float noise_seed,
    int num_neurons,
    cudaStream_t stream = nullptr
);

// Rotor Crystallization Fusion kernel
void launch_elysia_rotor_crystallization_fusion_kernel(
    const Rotor3D* d_psi_top,
    const Rotor3D* d_psi_bot,
    Rotor3D* d_rotors,
    float ach_level,
    float freeze_rate,
    float coherence_threshold,
    int num_neurons,
    cudaStream_t stream = nullptr
);

#ifdef __cplusplus
}
#endif

#endif // ELYSIA_FOC_KERNEL_H
