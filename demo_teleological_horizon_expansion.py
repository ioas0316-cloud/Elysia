"""
demo_teleological_horizon_expansion.py

Demonstration of ELYSIA's Embodied Teleological Horizon Expansion Runtime
==========================================================================
Showcasing:
1. Telos Purpose Kernel (0-Point Monad Attractor & Friction Transmutation into Growth Energy E_growth)
2. Dynamic Horizon Expansion Spectrometer (Escaping Fixed Container Trap: 3D -> 4D expansion on 'bitter' taste)
3. Body-Mind Topological Scale Continuity (Tactile/Motor Friction scaling to Conceptual Coordinate Axes)
"""

import numpy as np
from elysia_core.embodied_teleological_runtime import EmbodiedTeleologicalRuntime


def main():
    print("=========================================================================")
    print("  ELYSIA: Embodied Teleological Horizon Expansion Runtime Demo")
    print("=========================================================================\n")

    # Initialize runtime with 3 base taste axes: sweet, sour, salty
    runtime = EmbodiedTeleologicalRuntime(
        initial_dimension=3,
        expansion_threshold=0.35,
        friction_threshold=0.25,
        initial_axis_labels=["sweet", "sour", "salty"]
    )

    print(f"Initial Cognitive Dimension: {runtime.dimension}D")
    print(f"Initial Cognitive Axes     : {runtime.spectrometer.axis_labels}\n")

    # Stimulus Sequence
    stimuli = [
        ("sweet", np.array([1.0, 0.0, 0.0]), "light", "Pure Photon Sweet Emission"),
        ("sour", np.array([0.0, 1.0, 0.0]), "crayon", "Wax Drag Heavy Harmonics Sour"),
        ("salty", np.array([0.0, 0.0, 1.0]), "light", "Crystalline Wave Salty"),
        ("bitter", np.array([0.1, 0.1, 0.1, 1.2]), "pastel", "NOVEL: Powder Scattering Bitter Waveform"),
        ("bitter", np.array([0.1, 0.1, 0.1, 1.2]), "pastel", "Re-encounter Bitter Waveform")
    ]

    for idx, (label, vec, mech, desc) in enumerate(stimuli, 1):
        print(f"--- Step {idx}: Perceiving Stimulus [{label}] ({desc}) ---")
        summary = runtime.perceive_and_evolve(
            stimulus_vector=vec,
            stimulus_label_hint=label,
            mechanism=mech
        )

        print(f"  * Cognitive Friction (F_cog)   : {summary['cognitive_friction']:.4f}")
        print(f"  * Transmuted Growth Fuel (E_g) : {summary['growth_energy']:.4f}")
        print(f"  * Differential Tension (Delta) : {summary['differential_tension']:.4f}")

        if summary["horizon_expanded"]:
            print(f"  * >>> HORIZON EXPANSION TRIGGERED! (3D -> {summary['dimension']}D)")
            print(f"  * >>> Constructed New Coordinate Axis: [{summary['created_axis_label']}]")

        print(f"  * Current Active Axes         : {summary['active_axis_labels']}")
        print(f"  * Matched Axis Label          : {summary['matched_axis_label']}")
        print(f"  * Cellular Core Bifurcations  : {summary['total_bifurcated_cores']}")
        print(f"  * Telos Vector Norm           : {np.linalg.norm(summary['telos_combined_vector']):.4f}\n")

    print("=========================================================================")
    print("  Demo Complete: Telos Monad 0-Point Attractor successfully transmuted")
    print("  friction into growth fuel, escaping the fixed container trap via dynamic")
    print("  scale-free horizon expansion (3D -> 4D).")
    print("=========================================================================")


if __name__ == "__main__":
    main()
