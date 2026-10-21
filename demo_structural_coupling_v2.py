"""
Interactive Demo Script for ELYSIA Structural Coupling Runtime v2
Demonstrates the 3 Structural Coupling Principles:
1. Resonant Attractor Coupling (Conditionless Basin Convergence)
2. Meta-Plastic Prismatic Lens (Dynamic Refractive Matrix & Spectral Trajectory Feedback)
3. Cross-Modal Manifold Projection (Multimodal Field Fusion on Clifford Rotor Space)
"""

import numpy as np
import time
from elysia_core.structural_coupling_runtime import StructuralCouplingRuntime


def run_demo():
    print("=" * 80)
    print(" ELYSIA Structural Coupling & Meta-Plastic Engine v2 Demo")
    print("=" * 80)

    runtime = StructuralCouplingRuntime(target_dim=5)

    print("\n--- PHASE 1: Cross-Modal Input Synthesis & Initial Resonance ---")
    multimodal_inputs_1 = {
        "physical": np.array([1.0, 0.2, 0.0, 0.0, 0.0]),
        "sensory": np.array([0.8, 0.9, 0.1, 0.0, 0.0]),
        "chromatic": np.array([0.5, 0.3, 0.2]),
        "symbolic": np.array([0.1, 0.1, 0.8, 0.0, 0.0]),
    }

    freq_sequence = [1.0, 1.2, 1.1, 2.8, 3.1, 3.0, 3.9, 4.0, 1.0, 1.0]

    for step_idx, input_freq in enumerate(freq_sequence, 1):
        # Slightly alter physical input over time to simulate dynamic interaction
        multimodal_inputs_1["physical"] += np.random.uniform(-0.05, 0.05, 5)

        record = runtime.process_step(
            multimodal_inputs=multimodal_inputs_1,
            input_frequency=input_freq,
            dt=0.1
        )

        print(f"\n[Step {record['step']:02d}] Frequency: {input_freq:.1f} Hz")
        print(f"  └─ Cross-Modal Coherence   : {record['modal_coherence']:.4f}")
        print(f"  └─ Active Resonant Basin  : Basin #{record['active_basin_idx']} (Peak Resonance: {record['peak_resonance']:.4f})")
        print(f"  └─ Prismatic Refractive n : {record['prism_refractive_index']:.4f}")
        print(f"  └─ Macro Coherence Score   : {record['macro_coherence_score']:.4f}")
        print(f"  └─ Invariant Core Stability: {record['invariant_core_stability']:.4f}")

    print("\n" + "-" * 80)
    print("--- PHASE 2: Meta-Plastic Prism Lens Adaptation State ---")
    lens = runtime.prismatic_lens
    print(f"Refractive Index (n): {lens.refractive_index:.4f}")
    print("Prism Refraction Matrix (PrismMatrix) snapshot:")
    print(np.round(lens.prism_matrix, 3))

    if lens.spectral_history:
        print("\nTrajectory Spectral Fingerprint Power (First 5 components):")
        print(np.round(lens.spectral_history[-1][:5], 4))

    print("\n" + "=" * 80)
    print(" SUCCESS: ELYSIA Structural Coupling Runtime v2 Execution Complete!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
