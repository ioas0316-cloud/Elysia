"""
Demo: Transition from Brute-force Data Accumulation to Wisdom Abstraction & Generative Reconstruction
========================================================================================================
Demonstrates:
1. Absorbing ungrounded external data fragments into Causal Matrix.
2. Compressing repetitive micro-operations (1+1+1...) into $O(1)$ Executable Causal Formula.
3. Pruning obsolete / low-friction branches to prevent combinatorial explosion.
4. Generative Reconstruction of detailed micro-trajectories on demand.
5. Emitting Exhale Self-Explanation Pulse from digested principles.
"""

import numpy as np
from core.consciousness.causal_world_tree_engine import CausalWorldTreeEngine
from core.consciousness.causal_breathing_engine import ObserverTopology


def run_demo():
    print("=======================================================================")
    print("=== World Tree Engine: Data-to-Wisdom Abstraction & Respiration Demo ===")
    print("=======================================================================\n")

    engine = CausalWorldTreeEngine(critical_tension_threshold=8.0)

    # Step 1: Ingesting External Knowledge Fragment (Newton's Gravity)
    print("[1] Ingesting External Knowledge Fragment (외계 지식 들숨)...")
    res_gravity = engine.ingest_external_principle(
        principle_id="grav_law",
        name="Universal Gravitation Principle",
        domain="physics",
        raw_fragment="F = G * (m1 * m2) / r^2",
        invariant_vector=np.array([1.0, 0.5, 0.2, 0.8])
    )
    print(f"  -> Universal Stem Created: '{res_gravity['stem'].name}' (Wisdom Mass: {res_gravity['stem'].wisdom_mass:.2f})")
    print(f"  -> Causal Branch Grown: '{res_gravity['branch'].name}'")

    # Step 2: Compressing 10,000 Discrete Addition Steps into O(1) Executable Causal Formula
    print("\n[2] Executable Causal Formula Compression (실행형 인과수식 압축)...")
    formula = engine.compress_to_executable_formula(
        formula_id="formula_gravitational_pull",
        name="Gravitational Field Force Integrator",
        stem_id=res_gravity['stem'].stem_id,
        pattern_type="INVERSE_SQUARE_ACCUMULATION",
        discrete_step_count=10000
    )
    print(f"  -> Formula '{formula.name}' crystallized.")
    print(f"  -> Compression Ratio: {formula.compression_ratio:.0f}x discrete calculations saved!")

    # Evaluate Formula in O(1)
    eval_result = formula.evaluate({"n": 100.0, "multiplier": 1.5})
    print(f"  -> O(1) Evaluation Result: {np.round(eval_result, 3)}")

    # Step 3: Generative Reconstruction (생성적 역산 복원)
    print("\n[3] Generative Reconstruction on Demand (원리를 통한 궤적 역산 및 복원)...")
    trajectory = formula.generatively_reconstruct({"n": 100.0, "multiplier": 1.5}, detail_steps=4)
    for idx, step_coord in enumerate(trajectory):
        print(f"  -> Reconstructed Step {idx+1}: {np.round(step_coord, 3)}")

    # Step 4: Dynamic Synaptic Pruning (동적 가지치기 및 망각)
    print("\n[4] Dynamic Synaptic Pruning (조합 폭발 방지를 위한 동적 가지치기)...")
    # Grow obsolete branch under gravity stem
    from core.consciousness.causal_breathing_engine import MultiDimensionalAttractor
    obs_attractor = MultiDimensionalAttractor(
        id="att_obsolete_phlogiston",
        name="Phlogiston Friction Attractor",
        categorical_vector=np.array([0.1, 0.1, 0.1, 0.1]),
        sensorium_vector=np.array([0.1, 0.1, 0.1, 0.1]),
        morphology_vector=np.array([0.1, 0.1, 0.1, 0.1]),
        mass=0.02  # Low friction activity
    )
    b_obs = engine.grow_branch(
        branch_id="branch_obsolete_phlogiston",
        name="Phlogiston Friction Branch",
        stem_id=res_gravity['stem'].stem_id,
        attractor=obs_attractor,
        environmental_condition={"obsolete_friction": 0.01},
        depth=2
    )

    print(f"  -> Total branches before pruning: {len(engine.branches)}")
    pruned = engine.prune_unproductive_branches(activity_threshold=0.1, min_depth_to_keep=2)
    print(f"  -> Pruned Branches: {pruned}")
    print(f"  -> Total branches after pruning: {len(engine.branches)}")
    print(f"  -> Archived into Historical Ring: {len(engine.pruned_branches_archive)} branch(es)")

    # Step 5: Self-Explanation Respiration Pulse (세계수 날숨 서사 발산)
    print("\n[5] Respiration Exhale Pulse (세계수의 날숨 서사 발산)...")
    # Trigger threshold crossing via Inhale
    engine.inhale_world_stimulus(
        stimulus_id="stim_heavy_cosmic_pulse",
        categorical_vector=np.array([2.0, 2.0, 2.0, 2.0]),
        sensorium_vector=np.array([2.0, 2.0, 2.0, 2.0]),
        morphology_vector=np.array([2.0, 2.0, 2.0, 2.0]),
        reference_stem_id=res_gravity['stem'].stem_id,
        raw_description="High-density external gravity wave"
    )

    observer = ObserverTopology(observer_id="elysia_investigator", abstraction_capacity=0.9, causal_depth_tolerance=0.95)
    exhale_res, grand_narrative = engine.exhale_world_narrative(observer=observer)

    print(grand_narrative)
    print("\n=======================================================================")
    print("=== Demo Complete: Elysia World Tree has digested data into wisdom! ===")
    print("=======================================================================")


if __name__ == "__main__":
    run_demo()
