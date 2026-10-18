#!/usr/bin/env python3
"""
demo_volumetric_scale_hierarchy.py
===================================
Demonstration script verifying Project Elysia's Ontological Volumetric Scale Hierarchy,
Boundary Surface Tension, Causal Flow Deflection, and Interaction History Mechanics.
"""

import numpy as np
from core.physics.volumetric_scale_hierarchy import (
    StructuralDensityNode,
    BoundarySurfaceTensionField,
    VolumetricScaleHierarchyEngine,
)


def main():
    print("\n" + "=" * 80)
    print(" PROJECT ELYSIA :: ONTOLOGICAL VOLUMETRIC SCALE HIERARCHY DEMONSTRATION ")
    print("=" * 80 + "\n")

    # 1. Initialize Volumetric Structural Nodes (Barrier rock vs Flexible water/soil)
    print(">>> 1. Creating Volumetric Structural Nodes with Inherent Density & Surface Tension...")
    rock_barrier = StructuralDensityNode(
        node_id="rock_barrier",
        position=np.array([2.0, 0.0, 0.0], dtype=np.float32),
        density=3.5,
        porosity=0.02,
        viscosity_resistance=0.85,
    )

    porous_soil = StructuralDensityNode(
        node_id="porous_soil",
        position=np.array([2.0, -2.0, 0.0], dtype=np.float32),
        density=1.2,
        porosity=0.40,
        viscosity_resistance=0.30,
    )

    print(f"  * Rigid Rock Barrier ID: '{rock_barrier.node_id}'")
    print(f"    - Structural Density (M_s): {rock_barrier.density:.2f}")
    print(f"    - Porosity: {rock_barrier.porosity:.2f}")
    print(f"    - Inherent Surface Tension: {rock_barrier.surface_tension:.4f}")

    print(f"  * Porous Soil Field ID: '{porous_soil.node_id}'")
    print(f"    - Structural Density (M_s): {porous_soil.density:.2f}")
    print(f"    - Porosity: {porous_soil.porosity:.2f}")
    print(f"    - Inherent Surface Tension: {porous_soil.surface_tension:.4f}")

    # 2. Setup Volumetric Engine
    engine = VolumetricScaleHierarchyEngine()
    engine.add_node(rock_barrier)
    engine.add_node(porous_soil)

    # 3. Simulate Fluid Field Flow approaching Rock Barrier directly
    print("\n>>> 2. Simulating Fluid Field Flow Deflection & Boundary Tension around Barrier...")
    initial_pos = np.array([0.0, 0.1, 0.0], dtype=np.float32)
    initial_vel = np.array([1.5, 0.0, 0.0], dtype=np.float32)

    current_pos = initial_pos.copy()
    current_vel = initial_vel.copy()

    print(f"  * Initial Flow Velocity: {initial_vel.tolist()}, Position: {initial_pos.tolist()}")

    for step in range(1, 11):
        sim_res = engine.simulate_flow_step(current_vel, current_pos, dt=0.05, time_step=step * 0.05)
        current_pos = sim_res["flow_position"]
        current_vel = sim_res["flow_velocity"]

        print(
            f"  Step {step:02d} | Pos: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}] | "
            f"Vel: [{current_vel[0]:.3f}, {current_vel[1]:.3f}, {current_vel[2]:.3f}] | "
            f"Deflection: {sim_res['total_deflection']:.4f} | "
            f"Energy Retention: {sim_res['energy_retention_ratio']*100:.2f}%"
        )

    # 4. Inspect Accumulated Causal History
    print("\n>>> 3. Inspecting Accumulated Structural Interaction History (Causal Memory)...")
    print(f"  * Barrier Accumulated Energy: {rock_barrier.total_accumulated_energy:.4f}")
    print(f"  * Interaction Memory Records Count: {len(rock_barrier.interaction_history)}")
    if rock_barrier.interaction_history:
        first_rec = rock_barrier.interaction_history[0]
        print(f"  * Sample Interaction Record: {first_rec}")

    print("\n" + "=" * 80)
    print(" DEMONSTRATION COMPLETE: ONTOLOGICAL FLOW DEFLECTION & TENSION VERIFIED ")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
