#!/usr/bin/env python3
"""
[Demo] Dynamic Open Valence Phase Space Engine (+@)
Demonstrates:
1. Scalar state transition to continuous phase space trajectory (S_0, S_t, ΔS)
2. Open Latent Valence Vector (+@) interactions & field potential interference (-∇U_latent)
3. Emergent rupture detection and dynamic node spawning / instantiation
4. Alignment & Intrinsic Structural Valence tracking
"""

import numpy as np
from synaptic_architecture.phase_space_node import PhaseSpaceNode
from synaptic_architecture.open_valence_field import OpenValenceField
from synaptic_architecture.structural_valence import StructuralValence


def main():
    print("=" * 75)
    print("      ELYSIUS ARCHITECTURE: DYNAMIC OPEN VALENCE PHASE SPACE ENGINE (+@)")
    print("=" * 75)

    # 1. Initialize Open Valence Field & Structural Valence Engine
    field = OpenValenceField(rupture_threshold=2.8, max_nodes=20)
    structural_valence = StructuralValence(initial_dim=2, differentiation_threshold=2.0)

    print("\n[Step 1] Instantiating initial phase space nodes with open (+@) latent valence vectors...")

    # Node 1: "Good" / "Flow" state node floating in 2D phase space with 4D open latent valence
    n1 = field.create_node(
        position=np.array([0.0, 0.0]),
        latent_valence=np.array([1.5, 0.5, -0.2, 0.8]),
        velocity=np.array([0.1, 0.05]),
        core_potential=1.0,
        field_intensity=2.0,
        decay_gamma=0.3,
    )

    # Node 2: Approaching stimulus node with partially overlapping (+@) latent valence
    n2 = field.create_node(
        position=np.array([1.2, 0.5]),
        latent_valence=np.array([1.2, 0.8, 0.1, 0.5]),
        velocity=np.array([-0.2, -0.1]),
        core_potential=1.0,
        field_intensity=2.2,
        decay_gamma=0.3,
    )

    print(f"  Initialized Node 0 (S_0 = {n1.position}, +@ = {n1.latent_valence})")
    print(f"  Initialized Node 1 (S_0 = {n2.position}, +@ = {n2.latent_valence})")

    print("\n[Step 2] Running continuous phase field dynamics and potential interaction loop...")

    for step_idx in range(1, 15):
        step_info = field.step(dt=0.1, damping=0.05)

        # Evaluate structural valence for primary node (Node 0)
        n0 = field.nodes[0]
        val_eval = structural_valence.evaluate_valence(
            current_state=n0.position,
            current_velocity=n0.velocity,
            damped_friction=0.1,
            impedance=0.1,
            field_interference=n0.total_interference,
        )

        print(f"\n--- Frame {step_idx:02d} ---")
        print(f"  Active Nodes Count       : {step_info['active_nodes']}")
        print(f"  Node 0 Position (S_t)    : {np.round(n0.position, 3)}")
        print(f"  Node 0 Differential (ΔS) : {np.round(n0.velocity, 3)}")
        print(f"  Field Interference (U_tot): {n0.total_interference:.4f}")
        print(f"  Causal Tension Force     : {np.round(n0.tension_force, 4)}")
        print(f"  Evaluated Valence        : {val_eval['valence']:.4f} ({val_eval['state_label']})")

        if step_info["spawned_this_step"] > 0:
            print(f"  >>> RUPTURE DETECTED! Sponaneously spawned {step_info['spawned_this_step']} new node(s) <<<")
            for spawn in step_info["spawned_details"]:
                print(f"      Parent Node #{spawn['parent_id']} -> Child Node #{spawn['child_id']} at {np.round(spawn['spawn_position'], 3)}")

    print("\n" + "=" * 75)
    print("  SUMMARY OF AUTOPOIETIC PHASE SPACE DYNAMICS")
    print("=" * 75)
    print(f"  Total Nodes in Field     : {len(field.nodes)}")
    print(f"  Total Spawning History   : {len(field.spawn_history)} rupture event(s)")
    print(f"  Structural Categories    : {len(structural_valence.categories)}")
    print("=" * 75)
    print("Demonstration completed successfully.\n")


if __name__ == "__main__":
    main()
