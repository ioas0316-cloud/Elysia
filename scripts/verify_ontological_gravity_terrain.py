#!/usr/bin/env python3
"""
Verification Script: Ontological Gravity Terrain & Embodied Metanoia
Verifies smooth relaxation along continuous potential terrain towards Local Zero,
non-intrusive boundary deflection, and Inner vs. Outer World discrepancy self-calibration.
"""

import numpy as np
from core.physics.ontological_gravity_terrain_engine import OntologicalGravityTerrainEngine
from core.physics.causal_field import CausalField, InformationVoxel

def main():
    print("=" * 70)
    print("🌊 [Elysia] Ontological Gravity Terrain & Embodied Metanoia Verification")
    print("=" * 70)

    # 1. Initialize Ontological Gravity Terrain Engine
    engine = OntologicalGravityTerrainEngine(
        dimensions=16,
        anchor_stiffness=0.5,
        gravitational_constant=1.0,
        damping_factor=0.85
    )

    # Add a smooth, non-intrusive boundary mountain
    engine.add_boundary("mountain_1", center=np.array([2.0] * 16), amplitude=5.0, sigma=1.0)
    print(f"✅ Terrain initialized with 1 boundary obstacle (mountain_1) and stiffness {engine.anchor_stiffness}")

    # 2. Simulate Smooth Relaxation Flow towards Local Zero
    start_pos = np.array([5.0] * 16, dtype=np.float32)
    print(f"\n🌊 Simulating relaxation flow from initial position norm: {np.linalg.norm(start_pos):.4f}")

    trajectory = engine.simulate_relaxation(initial_position=start_pos, steps=30, dt=0.1)

    first = trajectory[0]
    mid = trajectory[len(trajectory)//2]
    last = trajectory[-1]

    print("  Step 00: Pos Norm = {:.4f} | Potential = {:.4f} | Gradient Norm = {:.4f}".format(
        np.linalg.norm(first.position), first.potential_energy, first.gradient_norm
    ))
    print("  Step 15: Pos Norm = {:.4f} | Potential = {:.4f} | Gradient Norm = {:.4f}".format(
        np.linalg.norm(mid.position), mid.potential_energy, mid.gradient_norm
    ))
    print("  Step 29: Pos Norm = {:.4f} | Potential = {:.4f} | Gradient Norm = {:.4f}".format(
        np.linalg.norm(last.position), last.potential_energy, last.gradient_norm
    ))

    assert np.linalg.norm(last.position) < np.linalg.norm(first.position), "Position did not move towards Local Zero!"
    assert last.potential_energy < first.potential_energy, "Potential energy did not decrease!"
    print("✅ Continuous relaxation along gravitational terrain verified!")

    # 3. Inner World vs Outer World Discrepancy Sensing & Metanoia Self-Calibration
    print("\n👁️ Simulating Inner World Thought vs Outer Reality Discrepancy Sensing (Metanoia)")
    inner_thought = np.array([4.0] * 16, dtype=np.float32)
    outer_reality = np.array([1.5] * 16, dtype=np.float32)

    prev_stiffness = engine.anchor_stiffness
    calib = engine.sense_and_recalibrate_discrepancy(inner_thought, outer_reality)

    print("  Discrepancy Vector Norm : {:.4f}".format(calib["discrepancy_norm"]))
    print("  Previous Anchor Stiffness: {:.4f}".format(calib["previous_stiffness"]))
    print("  New Anchor Stiffness     : {:.4f}".format(calib["new_stiffness"]))
    print("  Zero Self Shift Norm     : {:.4f}".format(calib["zero_self_shift"]))

    assert calib["new_stiffness"] > prev_stiffness, "Stiffness did not recalibrate!"
    print("✅ Metanoia self-calibration grounded in reality verified!")

    # 4. Integrated CausalField Verification
    print("\n⚙️ Testing Integration with CausalField")
    cf = CausalField(dimensions=3)
    voxel = InformationVoxel("v1", "Embodied Thought", np.ones(3), position=np.array([3.0, 3.0, 3.0]))
    cf.add_voxel(voxel)

    init_norm = np.linalg.norm(voxel.position)
    for _ in range(5):
        cf.step(0.1)

    post_norm = np.linalg.norm(voxel.position)
    print("  Initial Voxel Pos Norm: {:.4f} -> After 5 Field Steps Pos Norm: {:.4f}".format(init_norm, post_norm))
    assert post_norm < init_norm, "Voxel position did not relax in CausalField!"
    print("✅ Integrated CausalField gravitational relaxation verified!")

    print("\n" + "=" * 70)
    print("🎉 ALL ONTOLOGICAL GRAVITY TERRAIN VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
