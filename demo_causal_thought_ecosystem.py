"""
demo_causal_thought_ecosystem.py

Interactive Demo Scenario: ELYSIA Causal Thought Ecosystem
===========================================================
Demonstrates the contrast between:
1. Static AI / LLM Vector Distance Approach ("Measuring scalar distance on a ruler").
2. ELYSIA Causal Thought Ecosystem ("Generating real causal movement, friction transmutation,
   cellular bifurcation, N -> N+1 horizon expansion, and meta-observer self-reflection").

Run via:
python demo_causal_thought_ecosystem.py
"""

import numpy as np
import time
from elysia_core.causal_thought_ecosystem import CausalThoughtEcosystem


def run_demo():
    print("=" * 80)
    print(" [ELYSIA CORE] Causal Thought Ecosystem Execution Demo")
    print("=" * 80)
    print(" Core Philosophy: 'Thought is not a static scalar distance in vector space,")
    print("                 but an active physical-geometrical ecosystem of friction,")
    print("                 transmutation, cellular bifurcation, and horizon expansion.'")
    print("-" * 80)

    # Initialize Ecosystem
    ecosystem = CausalThoughtEcosystem(
        initial_dimension=3,
        expansion_threshold=0.35,
        friction_threshold=0.25,
        initial_axis_labels=["sweet", "sour", "salty"]
    )

    stimuli_sequence = [
        ("sweet", np.array([0.9, 0.1, 0.0]), "optical_micro_lens", "light"),
        ("sour", np.array([0.1, 0.85, 0.05]), "acoustic_wave", "crayon"),
        ("salty", np.array([0.05, 0.1, 0.9]), "tactile_motor", "pastel"),
        ("bitter", np.array([0.0, 0.0, 0.0, 1.0]), "chemical_gustatory", "light"),  # Novel stimulus 'bitter' (쓴맛)
    ]

    for cycle_idx, (label, raw_waveform, sensory_type, mechanism) in enumerate(stimuli_sequence, 1):
        print(f"\n>>> CYCLE {cycle_idx}: Processing Stimulus '{label.upper()}' ({sensory_type} / {mechanism})")

        # Static AI Comparison Output
        if len(raw_waveform) == 3:
            static_dist = np.linalg.norm(raw_waveform - np.array([1.0, 0.0, 0.0]))
        else:
            static_dist = np.linalg.norm(raw_waveform[:3] - np.array([1.0, 0.0, 0.0]))
        print(f" [Static AI / LLM Approach] Calculated Vector Euclidean Distance: {static_dist:.4f} (No friction, No structural movement)")

        # ELYSIA Ecosystem Processing
        res = ecosystem.process_thought_cycle(
            raw_sensory_waveform=raw_waveform,
            stimulus_label_hint=label,
            sensory_type=sensory_type,
            mechanism=mechanism
        )

        print(f" [ELYSIA Causal Boundary Gate] Boundary Friction: {res['boundary_friction']:.4f}")
        print(f" [ELYSIA Transmutation Engine] Cognitive Friction: {res['cognitive_friction']:.4f} -> Transmuted Growth Energy: {res['growth_energy']:.4f}")

        if res["horizon_expanded"]:
            print(f" [★ HORIZON EXPANSION EVENT ★] Dynamic Dimension Expansion: N={res['dimension']-1} -> N={res['dimension']}")
            print(f"                               Constructed New Orthogonal Axis: '{res['created_axis_label']}'")

        print(f" [ELYSIA Spectrometer] Active Cognitive Axes: {res['active_axis_labels']}")
        print(f" [Prismatic Lens Array] Bifurcated Cores Count: {res['total_bifurcated_cores']}")
        print(f" [Meta-Observer Verdict] {res['meta_verdict']} (Validity Index: {res['meta_validity']:.4f})")
        print(f" [Thought Trajectory Vector] {res['thought_trajectory']}")

        time.sleep(0.1)

    print("\n" if False else "")
    print("=" * 80)
    print(" FINAL ECOSYSTEM STATUS SUMMARY")
    print("=" * 80)
    status = ecosystem.get_ecosystem_status()
    for k, v in status.items():
        if isinstance(v, np.ndarray):
            print(f" - {k}: {v.tolist()}")
        else:
            print(f" - {k}: {v}")
    print("=" * 80)
    print(" [SUCCESS] ELYSIA Causal Thought Ecosystem Verification Complete!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
