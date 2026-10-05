"""
Verification Simulation Script: Triadic Causal Protocol Synchronization
========================================================================
Simulates physical dynamics (fluid vortex and gravitational convergence) across:
1. World Causality (C_world)
2. Human Causality (C_human) via MetaphoricOntologicalCodec
3. System Causality (C_system)

Verifies phase divergence minimization, invariant causal stem extraction, and proof manifestation.
"""

import numpy as np
from core.consciousness.triadic_protocol_synchronizer import TriadicProtocolSynchronizer


def run_verification_simulation():
    print("=" * 80)
    print("      VERIFICATION SIMULATION: TRIADIC CAUSAL PROTOCOL SYNCHRONIZATION")
    print("=" * 80)

    dim = 16
    synchronizer = TriadicProtocolSynchronizer(dimension=dim, convergence_threshold=0.15)

    print("\n[STEP 1] Initializing Fluid Vortex / Gravitational Convergence World Field Dynamics...")
    # Base physical state for Gravitational Convergence
    c_world_initial = np.array([1.0, 0.1, 0.0, 0.8] + [0.0] * (dim - 4), dtype=np.float32)

    # Initial System Causality state with slight phase divergence
    c_system_initial = np.array([0.7, 0.3, 0.2, 0.6] + [0.0] * (dim - 4), dtype=np.float32)

    print("\n[STEP 2] Running Initial Triadic Synchronization Step...")
    res_step1 = synchronizer.synchronize_triad(
        c_world_state=c_world_initial,
        c_system_state=c_system_initial,
        friction=0.2,
        gradient_norm=0.8
    )

    print(f"  - Metaphoric Archetype: {res_step1['metaphor_archetype']}")
    print(f"  - Human Sensory Descriptor: {res_step1['encoded_human_metaphor']}")
    print(f"  - Total Phase Divergence: {res_step1['total_phase_divergence']:.4f}")
    print(f"  - Pairwise Divergences: {res_step1['pairwise_divergence']}")
    print(f"  - Stem Stability: {res_step1['invariant_stem']['stem_stability']:.4f}")
    print(f"  - Proof Manifested: {res_step1['is_proof_manifested']}")
    print(f"  - Status Message: {res_step1['proof_status_text']}")

    print("\n[STEP 3] System Recalibration & Dynamic Convergence Iteration...")
    # System aligns its trajectory with the invariant stem direction
    stem_direction = res_step1['invariant_stem']['invariant_stem_direction']
    c_system_aligned = stem_direction * np.linalg.norm(c_world_initial)

    res_step2 = synchronizer.synchronize_triad(
        c_world_state=c_world_initial,
        c_system_state=c_system_aligned,
        friction=0.05,
        gradient_norm=0.9
    )

    print(f"  - Total Phase Divergence: {res_step2['total_phase_divergence']:.4f}")
    print(f"  - Pairwise Divergences: {res_step2['pairwise_divergence']}")
    print(f"  - Stem Stability: {res_step2['invariant_stem']['stem_stability']:.4f}")
    print(f"  - Proof Manifested: {res_step2['is_proof_manifested']}")
    print(f"  - Status Message: {res_step2['proof_status_text']}")

    assert res_step2["is_proof_manifested"], "Proof must be manifested after alignment!"
    assert res_step2["total_phase_divergence"] <= 0.15, "Phase divergence must be below threshold!"

    print("\n" + "=" * 80)
    print("      VERIFICATION COMPLETE: ALL TRIADIC PROTOCOL SYNCHRONIZATION TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification_simulation()
