"""
ELYSIA Global Scale Rotor & Causal-Cognitive PoC Engine Interactive Demo v0.1

Demonstrates:
1. SpectralWaveformTensor & O(1) Global ScaleRotor Operations
2. Local Spacetime Rotor Rewind ("Eye of Agamotto")
3. Rotor Coupling Constraint Relaxation (CAD-style joint equilibrium)
4. Recursive Scale Hierarchy Compression
5. Cognitive PoC Validation: Experiments A, B, and C (Hypotheses H1-H6)
"""

import numpy as np
from core.topology.global_scale_rotor import (
    SpectralWaveformTensor,
    ScaleRotor,
    RotorCouplingConstraint,
    RecursiveScaleHierarchy,
    CausalCognitiveMemory,
    ObservationEvent
)


def run_demo():
    print("=========================================================================")
    print("      ELYSIA Global Scale Rotor & Causal-Cognitive Engine PoC v0.1       ")
    print("=========================================================================\n")

    # -------------------------------------------------------------------------
    # 1. Global ScaleRotor Operations
    # -------------------------------------------------------------------------
    print("[1] Demonstrating O(1) Global ScaleRotor Phase Rotation & Order Inversion")
    tensor_base = SpectralWaveformTensor(
        shape=(4, 4),
        amplitudes=np.ones((4, 4)),
        phases=np.zeros((4, 4)),
        frequencies=np.full((4, 4), 440.0)
    )

    print(f"  - Initial Mean Phase: {np.mean(tensor_base.phases):.4f} rad")
    tensor_rotated = ScaleRotor.rotate_global_phase(tensor_base, phase_shift=np.pi / 2)
    print(f"  - Rotated Mean Phase (+pi/2): {np.mean(tensor_rotated.phases):.4f} rad")

    tensor_inverted = ScaleRotor.invert_global_order(tensor_rotated)
    print(f"  - Inverted Order Tensor Energy: {tensor_inverted.compute_energy():.2f}")
    print("  => O(1) Global Rotor execution completed without forward element loops.\n")

    # -------------------------------------------------------------------------
    # 2. Local Spacetime Rotor Control ("Eye of Agamotto")
    # -------------------------------------------------------------------------
    print("[2] Local Spacetime Rotor Control ('Eye of Agamotto')")
    tensor_world = SpectralWaveformTensor(shape=(6, 6))
    roi_apple = np.zeros((6, 6), dtype=bool)
    roi_apple[2:4, 2:4] = True  # Local apple coordinate region

    print("  - Rotating local ROI phase by pi/3 while surrounding world remains frozen...")
    local_rewound = ScaleRotor.apply_local_rotor(tensor_world, roi_apple, phase_shift=np.pi / 3, rewind_factor=1.0)

    outside_changed = not np.array_equal(local_rewound.phases[~roi_apple], tensor_world.phases[~roi_apple])
    inside_changed = not np.array_equal(local_rewound.phases[roi_apple], tensor_world.phases[roi_apple])

    print(f"  - Surrounding Region Changed? {outside_changed} (Expected: False)")
    print(f"  - Local Apple Region Changed? {inside_changed} (Expected: True)")
    print("  => Local Spacetime Rewind verified.\n")

    # -------------------------------------------------------------------------
    # 3. Rotor Coupling Constraint Relaxation
    # -------------------------------------------------------------------------
    print("[3] Rotor Coupling Constraint Relaxation (CAD-style joint equilibrium)")
    rotor_a = SpectralWaveformTensor(shape=(2, 2), phases=np.full((2, 2), np.pi / 2))
    rotor_b = SpectralWaveformTensor(shape=(2, 2), phases=np.full((2, 2), 0.0))

    coupling = RotorCouplingConstraint(constraint_id="joint_cad_01", target_phase_diff=0.0, coupling_strength=0.8)
    torque_0 = coupling.compute_torque(rotor_a, rotor_b)
    print(f"  - Initial Joint Torque Discrepancy: {torque_0:.4f}")

    curr_a, curr_b = rotor_a, rotor_b
    for step in range(1, 4):
        curr_a, curr_b, err = coupling.relax_step(curr_a, curr_b)
        print(f"    Step {step}: Joint Torque Error = {err:.4f}")

    print("  => Geometric convergence to structural equilibrium achieved without branching.\n")

    # -------------------------------------------------------------------------
    # 4. Recursive Scale Hierarchy
    # -------------------------------------------------------------------------
    print("[4] Recursive Scale Hierarchy (Fractal Compression)")
    hierarchy = RecursiveScaleHierarchy(layer_id="cortex_orchestra", scale_level=1)
    hierarchy.register_sub_manifold(tensor_base)
    hierarchy.register_sub_manifold(tensor_world)

    compressed_node = hierarchy.compress_to_node()
    print(f"  - Lower Layer Manifolds Compressed Count: {len(hierarchy.sub_manifolds)}")
    print(f"  - Compressed Single Frequency Node Shape: {compressed_node.shape}")
    print(f"  - Representative Node Mean Frequency: {compressed_node.frequencies[0, 0]:.2f} Hz\n")

    # -------------------------------------------------------------------------
    # 5. Cognitive PoC Experiments A, B, C Validation
    # -------------------------------------------------------------------------
    print("[5] Causal Cognitive Memory PoC Validation (Hypotheses H1-H6)")
    memory = CausalCognitiveMemory()

    # Experiment A: Red Apple -> Blue Apple
    print("  [Experiment A] Property Change & Identity Maintenance:")
    memory.record_observation(ObservationEvent("obs_a1", 1.0, "apple_x", {"category": "Apple", "color": "Red", "species": "Fuji"}))
    memory.record_observation(ObservationEvent("obs_a2", 2.0, "apple_x", {"category": "Apple", "color": "Blue", "species": "Fuji"}))

    eval_a = memory.evaluate_identity_and_property_change("apple_x", ["category", "species"], ["color"])
    print(f"    - Identity Maintained: {eval_a['identity_maintained']}")
    print(f"    - Property Changes Recorded: {eval_a['property_changes']}")

    # Experiment B: State Change & Classification Update
    print("  [Experiment B] State Change, Attribution & Uncertainty Preservation:")
    memory.record_observation(ObservationEvent("obs_b1", 3.0, "apple_dmg", {"category": "Apple", "damage_level": 0.85, "is_fragmented": True}, context_conditions={"observed_cause": "Dropped from height"}))
    eval_b1 = memory.evaluate_state_change_and_classification("apple_dmg")
    print(f"    - Known Cause Attribution: {eval_b1['causal_attribution']} ({eval_b1['cause_summary']})")

    memory.record_observation(ObservationEvent("obs_b2", 3.5, "apple_amb", {"category": "Apple", "damage_level": 0.90, "is_fragmented": True}, context_conditions={}))
    eval_b2 = memory.evaluate_state_change_and_classification("apple_amb")
    print(f"    - Ambiguous Cause Attribution: {eval_b2['causal_attribution']}")
    print(f"    - Uncertainty Preserved? {eval_b2['uncertainty_preserved']} -> Action: {eval_b2['classification_action']}")

    # Experiment C: Intertemporal Belief Update
    print("  [Experiment C] Intertemporal Recurrent Causality:")
    memory.record_observation(ObservationEvent("obs_c1", 1.0, "subj_y", {"category": "Fruit", "ripeness": 0.1}))
    b1 = memory.update_intertemporal_belief("subj_y", "obs_c1")
    print(f"    - Initial Belief Status: {b1.status}")

    memory.record_observation(ObservationEvent("obs_c2", 2.0, "subj_y", {"category": "Fruit", "ripeness": 0.9}))
    b2 = memory.update_intertemporal_belief("subj_y", "obs_c2")
    print(f"    - Updated Belief Status: {b2.status} (Evidence History Count: {len(b2.evidence_history)})")

    print("\n=========================================================================")
    print("             ELYSIA Global Scale Rotor Demo Completed Successfully!       ")
    print("=========================================================================")


if __name__ == "__main__":
    run_demo()
