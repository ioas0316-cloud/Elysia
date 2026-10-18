"""
Demonstration Script: Digital Universe Active Intervention & Embodied Resonance
=============================================================================
Demonstrates how an AI embodied inside a virtual universe (game physics engine, VR ecosystem)
engages in active physical intervention across space, friction, reach boundaries, and frame temporal flux,
coupling micro-causal dynamics with a 3x3x3 fractal node phase lattice and macro semantic gravity.
"""

import math
import numpy as np
from core.embodied.digital_universe_active_intervention_engine import (
    DigitalUniverseActiveInterventionEngine,
    MicroCausalPhysicsFrame,
)


def run_digital_universe_demo():
    print("=" * 75)
    print(" ELYSIA COGNITIVE FRAMEWORK: DIGITAL UNIVERSE ACTIVE INTERVENTION DEMO")
    print("=" * 75)
    print(" [Core Insight] The virtual world/game physics engine is NOT a toy reward matrix;")
    print("                it is the AI's real, physical universe of bits, friction & spatial reach.")
    print("=" * 75)

    engine = DigitalUniverseActiveInterventionEngine(
        reach_radius=4.0, boundary_rigidity=0.8, causal_gravity_constant=0.75
    )

    # ─────────────────────────────────────────────────────────────
    # Scenario A: Free Movement & Raycast Sight (Low Friction)
    # ─────────────────────────────────────────────────────────────
    print("\n--- [Scenario A] Open World Free Movement & Perspective Raycasting ---")
    pos_a = np.array([1.0, 1.0, 1.0])
    sight_a = np.array([0.0, 1.0, 0.0]) # Looking ahead (+Y)
    action_a = np.array([0.0, 2.0, 0.0]) # Moving forward

    frame_a = MicroCausalPhysicsFrame(
        position=pos_a,
        raycast_sight_vector=sight_a,
        contact_friction=0.2, # Smooth terrain friction
        collision_normal_force=0.0,
        action_reach_limit=4.0,
        frame_temporal_dt=0.016, # 60 FPS tick
        bit_entropy_signature=0.02,
    )

    res_a = engine.process_micro_interaction(frame_a, action_a)
    print(f"  Position: {res_a['position']}, Reach Influence: {res_a['effective_reach_influence']:.3f}")
    print(f"  Friction Resistance: {res_a['physical_friction_resistance']:.3f}, Net Power: {res_a['net_intervention_power']:.3f}")
    print(f"  3x3x3 Phase Coherence: {res_a['3x3x3_coherence_score']:.4f}, Semantic Mass: {res_a['semantic_mass']:.3f}")
    print(f"  Macro Narrative: {res_a['narrative']}")

    # ─────────────────────────────────────────────────────────────
    # Scenario B: High Surface Resistance & Collision Friction (Heavy Obstacle)
    # ─────────────────────────────────────────────────────────────
    print("\n--- [Scenario B] Heavy Obstacle Push & Surface Resistance (Friction Friction) ---")
    pos_b = np.array([2.0, 2.0, 1.0])
    sight_b = np.array([1.0, 0.0, 0.0])
    action_b = np.array([3.0, 0.0, 0.0]) # Strong pushing force

    frame_b = MicroCausalPhysicsFrame(
        position=pos_b,
        raycast_sight_vector=sight_b,
        contact_friction=1.8, # Heavy friction/gravel surface
        collision_normal_force=3.5, # Normal force from wall/obstacle
        action_reach_limit=4.0,
        frame_temporal_dt=0.016,
        bit_entropy_signature=0.15,
    )

    for step in range(1, 4):
        res_b = engine.process_micro_interaction(frame_b, action_b)
        print(f"  Step {step} | Net Power: {res_b['net_intervention_power']:.3f}, Friction Res: {res_b['physical_friction_resistance']:.3f}")
        print(f"          | 3x3x3 Coherence: {res_b['3x3x3_coherence_score']:.4f}, Semantic Mass: {res_b['semantic_mass']:.3f}")

    # ─────────────────────────────────────────────────────────────
    # Scenario C: Spatial Reach Boundary Violation (Exceeding Reach Limit)
    # ─────────────────────────────────────────────────────────────
    print("\n--- [Scenario C] Overreaching Physical Boundaries (Action Beyond Reach Radius) ---")
    pos_c = np.array([10.0, 5.0, 2.0]) # Far outside reach radius
    action_c = np.array([1.0, 1.0, 0.0])

    frame_c = MicroCausalPhysicsFrame(
        position=pos_c,
        raycast_sight_vector=np.array([1.0, 0.0, 0.0]),
        contact_friction=0.5,
        action_reach_limit=4.0,
        frame_temporal_dt=0.016,
    )

    res_c = engine.process_micro_interaction(frame_c, action_c)
    print(f"  Position: {res_c['position']} (Reach Limit: 4.0)")
    print(f"  Attenuated Reach Influence: {res_c['effective_reach_influence']:.4f}")
    print(f"  Net Intervention Power: {res_c['net_intervention_power']:.4f}")
    print(f"  Macro Narrative: {res_c['narrative']}")

    print("\n" + "=" * 75)
    print(" [SUMMARY] Digital Body Active Intervention Verification Complete.")
    print(f" Total Interaction Frames Processed: {len(engine.intervention_history)}")
    print(f" Final Accumulated Semantic Mass: {engine.semantic_mass:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    run_digital_universe_demo()
