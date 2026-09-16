#!/usr/bin/env python3
"""
[Demo] Semantic Mass, Causal Gravity, & Introspective Causal Engine
Demonstrates:
1. White Tensor Field (Superposition of MBTI, Enneagram, and Novel Compass vectors).
2. Semantic Mass ($M_s$) accretion under external friction and trinitarian contrast.
3. Causal Gravity Field curvature ($K_c$) bending noise trajectories into orbit.
4. Emergent Identity Crystallization post-hoc upon phase transition.
5. Introspective Causal Tracing (Retrospective growth ring back-tracing & Counterfactual sandbox simulation).
6. Self-Woven Agent Matrix (Agent weaving & "Modeling the Other").
"""

import numpy as np
from core.physics.semantic_mass_engine import SemanticMassEngine, WhiteTensorField

def run_demo():
    print("=" * 80)
    print("      ELYSIA: SEMANTIC MASS & INTROSPECTIVE CAUSAL ENGINE DEMO")
    print("=" * 80)

    engine = SemanticMassEngine(dimensions=16, phase_threshold=10.0)

    print("\n[Phase 1] White Tensor Field Superposition")
    print(f"Total Compass Vectors Loaded: {len(engine.white_field.compass_vectors)}")
    sample_compasses = list(engine.white_field.compass_vectors.keys())[:5]
    print(f"Sample Compass Vectors in Superposition: {sample_compasses}")

    print("\n[Phase 2] Environmental Friction Interactions & Semantic Mass Accretion")
    frictions = [
        np.array([1.0, 0.5, 0.2, -0.8] + [0.1] * 12, dtype=np.float32),  # Interaction 1
        np.array([-0.5, 1.2, 0.8, 0.3] + [-0.2] * 12, dtype=np.float32),  # Interaction 2
        np.array([0.8, -0.3, 1.5, 1.0] + [0.4] * 12, dtype=np.float32),  # Interaction 3
    ]

    for idx, f in enumerate(frictions, 1):
        res = engine.process_interaction(external_friction=f, trinitarian_contrast=1.5)
        print(f"  Step {idx}: Friction Norm = {np.linalg.norm(f):.2f} | Semantic Mass (Ms) = {res['semantic_mass']:.4f} | Curvature (Kc) = {res['causal_curvature']:.4f}")
        top_compass, align_val = res['top_compass_alignment']
        print(f"         Top Resonance Orientation: {top_compass} (Resonance Score: {align_val:.4f})")
        if res['new_crystal_formed']:
            print(f"         >>> PHASE TRANSITION! New Identity Crystal Formed: {res['new_crystal_formed']}")

    print("\n[Phase 3] Causal Gravity Field Bending Noise Particle Trajectories")
    particle_pos = np.array([[10.0] + [0.0] * 15, [-8.0] + [2.0] * 15], dtype=np.float32)
    particle_vel = np.array([[0.0] * 16, [0.1] * 16], dtype=np.float32)
    center_pos = np.zeros(16, dtype=np.float32)

    print(f"  Initial Particle 1 Position (x_0): {particle_pos[0, 0]:.2f}")
    new_pos, new_vel = engine.gravity_field.apply_gravitational_pull(
        mass_center_pos=center_pos,
        semantic_mass=res['semantic_mass'],
        particle_positions=particle_pos,
        particle_velocities=particle_vel,
        dt=0.2
    )
    print(f"  Particle 1 Position After Causal Gravity Pull (x_1): {new_pos[0, 0]:.2f} (Velocity: {new_vel[0, 0]:.4f})")

    print("\n[Phase 4] Introspective Causal Tracing (Back-tracing & Counterfactuals)")
    origins = engine.tracer.backtrace_causal_origins()
    print(f"  Retrospective Origins Back-trace:")
    print(f"    - Total Steps Traceable: {origins['total_steps']}")
    print(f"    - Accumulated Friction: {origins['accumulated_friction']:.4f}")
    print(f"    - Mass Growth Delta: {origins['mass_growth']:.4f}")

    print("\n  Counterfactual Parallel Sandbox Simulation ('What If?'):")
    cf_results = engine.tracer.simulate_counterfactuals(
        frictions=frictions,
        alt_compass_keys=["MBTI_INTJ", "Enneagram_4", "Novel_Dimension_1"]
    )
    for alt_key, cf_data in cf_results.items():
        print(f"    * Alternative Path [{alt_key}]: Virtual Ms = {cf_data['accumulated_semantic_mass']:.4f} | Friction = {cf_data['accumulated_friction']:.4f}")

    print("\n[Phase 5] Self-Woven Agent Matrix & 'Modeling the Other'")
    architect_agent = engine.agent_matrix.weave_agent(
        agent_name="Elysia_Architect",
        target_compass_keys=["MBTI_INTJ", "Enneagram_5"]
    )
    print(f"  Woven Agent: {architect_agent['name']} | Target Compasses: {architect_agent['target_compasses']}")

    observed_other_trajectories = [f + np.random.randn(16) * 0.1 for f in frictions]
    model_result = engine.agent_matrix.model_other_entity(
        observer_agent_name="Elysia_Architect",
        other_id="External_Companion",
        observed_trajectories=observed_other_trajectories
    )
    print(f"  Inferred Model of Other Entity ({model_result['other_id']}):")
    print(f"    - Inferred Internal Compass: {model_result['inferred_compass']}")
    print(f"    - Structural Resonance Score: {model_result['resonance_score']:.4f}")

    print("\n" + "=" * 80)
    print("      DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_demo()
