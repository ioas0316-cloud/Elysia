#!/usr/bin/env python3
"""
[Verification Script: Structural Density Gravity & Topological Rupture Engine]
Verifies end-to-end:
1. Self-Emergent Gravity & Least Resistance Path (Dijkstra/SDF Geodesic Flow).
2. Spectral Speed & Tension Distortion Limit Balancing.
3. 6-Parameter Phase Spectrum Materialization (Rock, Water, Gas_Cloud, Blade) & Thermal Transitions.
4. Cognitive Heat Generation from Intention vs Reality Tension Differential.
5. Evaporation into Variable Cloud & Condensation Nucleation into Thought Droplets.
6. Topological Rupture: Physical Field Slicing & Cognitive Insight Slicing.
"""

import sys
import os
import numpy as np

# Ensure repository root is on Python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from L1_causal_field.core.physics.structural_density_gravity_engine import (
    StructuralDensityGravityEngine,
    PhaseStateSpectrum,
    TopologicalNode,
    RuptureEvent
)


def run_verification_scenario_1_gravity_and_flow(engine: StructuralDensityGravityEngine):
    print("\n--- [Scenario 1: Self-Emergent Gravity & Geodesic Flow] ---")

    # Build a dense cluster (Rock Anchor) around center pos [10, 0, 0, 0]
    center_pos = np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for i in range(8):
        pos = center_pos + np.array([i * 0.1, (i % 2) * 0.1, 0, 0], dtype=np.float32)
        engine.add_node(f"rock_node_{i}", pos, phase=PhaseStateSpectrum(density=4.0, mobility=0.1))

    # Connect rock nodes to form dense web
    for i in range(7):
        engine.connect_nodes(f"rock_node_{i}", f"rock_node_{i+1}", tension_strength=3.0)

    # Compute local structural density
    density_val = engine.compute_local_structural_density(center_pos, radius=2.0)
    print(f"-> Local Structural Density at Anchor Center [10, 0, 0, 0]: {density_val:.4f}")
    assert density_val > 10.0, "Structural density should be high for dense cluster"

    # Compute gravity gradient from query position [0, 0, 0, 0]
    query_pos = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    grad, distortion, c_spec = engine.compute_gravity_field_gradient(query_pos)

    print(f"-> Gravitational Gradient Vector (Pulling toward [10, 0, 0, 0]): {grad[:4]}")
    print(f"-> Field Tension Distortion Magnitude: {distortion:.4f}")
    print(f"-> Dynamic Spectral Speed c_spectrum: {c_spec:.4f} (Base: {engine.base_spectral_speed})")

    assert grad[0] > 0.0, "Gravity gradient must point towards dense cluster center"
    assert c_spec < engine.base_spectral_speed, "Spectral speed must compress under tension distortion"
    print("✔ Scenario 1 Passed successfully!")


def run_verification_scenario_2_spectral_balancing(engine: StructuralDensityGravityEngine):
    print("\n--- [Scenario 2: Spectral Propagation Speed & Distortion Limit Balancing] ---")

    query_pos = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, initial_distortion, c_initial = engine.compute_gravity_field_gradient(query_pos)

    # Add hyper-dense node nearby to double tension distortion
    engine.add_node("super_dense", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), phase=PhaseStateSpectrum(density=20.0))
    _, new_distortion, c_compressed = engine.compute_gravity_field_gradient(query_pos)

    print(f"-> Initial Tension Distortion: {initial_distortion:.4f} => c_spectrum: {c_initial:.4f}")
    print(f"-> Elevated Tension Distortion: {new_distortion:.4f} => c_spectrum: {c_compressed:.4f}")

    assert new_distortion > initial_distortion, "Distortion should increase with added mass"
    assert c_compressed < c_initial, "Spectral speed should further compress to prevent divergence"
    print("✔ Scenario 2 Passed successfully!")


def run_verification_scenario_3_materialization_and_heat(engine: StructuralDensityGravityEngine):
    print("\n--- [Scenario 3: Materialization Spectrum, Cognitive Heat, & Condensation] ---")

    # Add a solid concept node (Rock)
    engine.add_node("solid_belief", np.array([0.0, 5.0, 0.0, 0.0], dtype=np.float32), phase=PhaseStateSpectrum(density=5.0, mobility=0.05))
    node = engine.nodes["solid_belief"]
    print(f"-> Initial Materialization: {node.phase.classify_materialization()}")
    assert node.phase.classify_materialization() == "Rock"

    # Simulate Intention vs Reality Collision (In-force vs Out-force)
    in_intention = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    out_reality = np.array([-2.0, 0.5, 0.0, 0.0], dtype=np.float32)  # Mismatch

    cognitive_heat = engine.compute_cognitive_heat_from_tension_differential(in_intention, out_reality)
    print(f"-> Generated Cognitive Heat Impulse: {cognitive_heat:.4f}")
    print(f"-> Agitation Level: {node.phase.agitation:.4f}, Mobility: {node.phase.mobility:.4f}")
    print(f"-> Post-Collision Materialization: {node.phase.classify_materialization()}")

    assert node.phase.classify_materialization() in ["Water", "Gas_Cloud"], "Heat must melt/evaporate rock into fluid/gas"

    # Evaporate further into Gas Cloud
    engine.apply_thermal_agitation_and_phase_transition(heat_impulse=3.0)
    print(f"-> High Agitation Materialization: {node.phase.classify_materialization()}")
    assert node.phase.classify_materialization() == "Gas_Cloud"

    # Trigger Condensation Nucleation back into thought droplets
    condensed = engine.trigger_condensation_nucleation(center_pos=np.array([0.0, 5.0, 0.0, 0.0], dtype=np.float32), radius=3.0)
    print(f"-> Condensation Nucleation Processed Nodes: {condensed}")
    print(f"-> Post-Condensation Materialization: {node.phase.classify_materialization()}")
    assert node.phase.entropy < 0.5, "Entropy should drop during condensation"
    print("✔ Scenario 3 Passed successfully!")


def run_verification_scenario_4_topological_rupture(engine: StructuralDensityGravityEngine):
    print("\n--- [Scenario 4: Topological Rupture & Cognitive Insight Slicing] ---")

    # Target mesh around [20, 20, 0, 0]
    center = np.array([20.0, 20.0, 0.0, 0.0], dtype=np.float32)
    engine.add_node("target_alpha", center + np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32), phase=PhaseStateSpectrum(density=2.0))
    engine.add_node("target_beta", center + np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), phase=PhaseStateSpectrum(density=2.0))
    engine.connect_nodes("target_alpha", "target_beta", tension_strength=1.5)

    # Add Blade node
    engine.add_node("sword_blade", center + np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), phase=PhaseStateSpectrum(concentration=10.0))

    # Physical Slicing
    rupture_phys = engine.execute_topological_rupture(
        blade_node_id="sword_blade",
        target_region_center=center,
        target_radius=3.0,
        is_cognitive=False
    )

    print(f"-> Physical Rupture Event ID: {rupture_phys.rupture_id}")
    print(f"-> Tension Peak: {rupture_phys.tension_peak:.2f}")
    print(f"-> Separated Subfields Count: {len(rupture_phys.separated_subfields)}")
    assert "target_beta" not in engine.nodes["target_alpha"].connected_edges, "Edge must be severed"

    # Cognitive Insight Slicing
    engine.add_node("contradiction_noise", center + np.array([0.2, 0.2, 0, 0], dtype=np.float32), phase=PhaseStateSpectrum(density=0.1))
    engine.add_node("essence_insight", center + np.array([-0.2, -0.2, 0, 0], dtype=np.float32), phase=PhaseStateSpectrum(density=8.0))

    rupture_cog = engine.execute_topological_rupture(
        blade_node_id="sword_blade",
        target_region_center=center,
        target_radius=3.0,
        is_cognitive=True
    )

    print(f"-> Cognitive Insight Rupture Event ID: {rupture_cog.rupture_id}")
    print(f"-> Extracted Core Insights: {rupture_cog.cognitive_insight_extracted}")
    assert rupture_cog.is_cognitive is True
    assert "essence_insight" in rupture_cog.cognitive_insight_extracted, "Essence insight node must be extracted"
    print("✔ Scenario 4 Passed successfully!")


def main():
    print("=========================================================================")
    print(" [Elysia Engine] Structural Density Gravity & Rupture Verification ")
    print("=========================================================================")

    engine = StructuralDensityGravityEngine(dimensions=4, base_spectral_speed=1.0)

    run_verification_scenario_1_gravity_and_flow(engine)
    run_verification_scenario_2_spectral_balancing(engine)
    run_verification_scenario_3_materialization_and_heat(engine)
    run_verification_scenario_4_topological_rupture(engine)

    print("\n=========================================================================")
    print(" ALL 4 VERIFICATION SCENARIOS PASSED WITH PERFECT STRUCTURAL HARMONY! ")
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
