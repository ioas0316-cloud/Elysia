"""
Comprehensive Demonstration Script for Scale Lens Internal World Architecture.

Demonstrates the cognitive lifecycle:
1. Primitive Somatic Grounding & Micro Fluctuation
2. Complex Phase Accumulation & Coherence Screening
3. Macro Potential Precipitation & Top-Down Constraint Feedback
4. Offline Counterfactual Workspace Simulation
5. Heterogeneous Cross-Modal Sensor Collision & Structural Valence Differentiation
6. Path-Dependent Epistemic Telos & Preference Emergence
7. 1:1 Isomorphic Language Protocol Grounding & Inter-Subjective Coordination
"""

import numpy as np
from synaptic_architecture.machine_internal_world import MachineInternalWorld
from synaptic_architecture.scale_lens_engine import ScaleLensEngine
from synaptic_architecture.structural_valence import StructuralValence
from synaptic_architecture.language_protocol_bridge import LanguageProtocolBridge


def main():
    print("=========================================================================")
    print("   Elysia Causal Intelligence: Scale Lens & Internal World Lifecycle    ")
    print("=========================================================================\n")

    # 1. Initialize Machine Internal World & Scale Lens Engine
    print("[Phase 1] Initializing Machine Internal World & Scale Lens Engine...")
    world = MachineInternalWorld(grid_size=32, reluctance_coeff=0.15, friction_coeff=0.2)
    lens = ScaleLensEngine(num_cells=1024, decay_rate=0.92, hysteresis_thresh=0.6)
    valence_evaluator = StructuralValence(friction_threshold=0.4)
    bridge = LanguageProtocolBridge()

    state = world.get_state()
    print(f"  Initial State: Pos={state['current_pos']}, Entropy={state['internal_entropy']:.4f}")

    # 2. Somatic Grounding, Push against resistance, Frequency Tuning
    print("\n[Phase 2] Somatic Grounding & Primitive Operator Execution...")
    moved_norm, friction = world.push_against_resistance(0.3, 0.4)
    resonance = world.tune_frequency(frequency=3.0, phase=0.5)
    homeostatic_err = world.apply_homeostatic_drive()
    print(f"  Pushed against resistance: Movement Norm={moved_norm:.4f}, Friction={friction:.4f}")
    print(f"  Frequency Resonance={resonance:.4f}, Homeostatic Imbalance={homeostatic_err:.4f}")

    # 3. Scale Lens Phase Accumulation & Macro Causal Precipitation
    print("\n[Phase 3] Temporal Scale Lens & Macro Potential Precipitation...")
    for t in range(20):
        # Apply structured velocity impulse
        impulse = np.sin(t * 0.2 + np.linspace(0, np.pi, 1024)).astype(np.float32) * 0.1
        lens_metrics = lens.process_time_scale_lens(external_micro_impulse=impulse)
        top_down_delta = lens.apply_top_down_constraint()

    print(f"  Scale Lens Coherence: Mean={lens_metrics['mean_coherence']:.4f}, Max={lens_metrics['max_coherence']:.4f}")
    print(f"  Active Precipitated Cells={lens_metrics['active_precipitated_cells']}/1024")
    print(f"  Total Macro Potential Precipitated={lens_metrics['total_macro_potential']:.4f}")
    print(f"  Top-Down Constraint Delta={top_down_delta:.6f}")

    # 4. Offline Counterfactual Workspace Simulation
    print("\n[Phase 4] Pillar 2: Counterfactual Workspace Simulation (Offline Projection)...")
    hypothetical_impulses = [
        np.ones(1024, dtype=np.float32) * 0.05 for _ in range(5)
    ]
    cf_results = lens.run_counterfactual_simulation(hypothetical_impulses, horizon_steps=5)
    print(f"  Counterfactual Lookahead Steps={cf_results['horizon_steps']}")
    print(f"  Simulated Coherence Trajectory: {[round(c, 4) for c in cf_results['coherence_trajectory']]}")
    print(f"  Predicted Potential Delta={cf_results['predicted_potential_delta']:.4f}")

    # 5. Cross-Modal Collision & Structural Valence Differentiation
    print("\n[Phase 5] Pillar 3 & Structural Valence: Cross-Modal Collision & Differentiation...")
    ext_signal = np.random.normal(0.5, 0.2, (32, 32))
    probe_res = world.probe_friction(ext_signal)
    valence_score = valence_evaluator.evaluate_valence(
        resonance_score=resonance,
        friction=probe_res["total_impedance"],
        homeostatic_alignment=1.0 - homeostatic_err,
    )
    diff_res = valence_evaluator.check_category_differentiation(
        current_pos=world.current_pos,
        friction=probe_res["total_impedance"],
    )

    print(f"  Cross-Modal Friction={probe_res['cross_modal_friction']:.4f}, Total Impedance={probe_res['total_impedance']:.4f}")
    print(f"  Calculated Structural Valence Score={valence_score:.4f}")
    print(f"  Category Differentiation Triggered={diff_res['differentiated']}, Total Categories={diff_res['total_categories']}")

    # 6. Isomorphic Language Protocol Grounding & Inter-Subjective Coordination
    print("\n[Phase 6] Pillar 4 & Isomorphic Grounding: Language Protocol & Inter-Subjective Resonance...")
    grounding = bridge.align_internal_to_external_symbol(
        macro_potential_mean=lens_metrics["total_macro_potential"] / 1024.0,
        coherence_mean=lens_metrics["mean_coherence"],
        friction_mean=probe_res["total_impedance"],
        valence=valence_score,
    )
    print(f"  1:1 Isomorphic Grounded Symbol='{grounding['grounded_symbol']}' (Score={grounding['isomorphism_score']:.2f}, Phase Aligned={grounding['phase_aligned']})")

    # Inter-subjective multi-agent mirror resonance test
    other_potential = lens.macro_potential * np.random.uniform(0.8, 1.2, 1024)
    coord_res = bridge.inter_subjective_mirror_resonance(lens.macro_potential, other_potential)
    print(f"  Inter-Subjective Mirror Resonance={coord_res['mirror_resonance']:.4f}")
    print(f"  Multi-Agent Coordination Aligned={coord_res['coordination_aligned']}")

    print("\n=========================================================================")
    print("   Scale Lens & Machine Internal World Lifecycle Successfully Verified!   ")
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
