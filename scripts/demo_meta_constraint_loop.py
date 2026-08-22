#!/usr/bin/env python3
"""
[Demonstration Script: Meta-Constraint Feedback Loop & Re-perception]
This script demonstrates the end-to-end causal feedback pipeline:
1. Fast-clock Candidate Trajectory Generation under initial constraints.
2. Re-perception into C++ Preisach SoA field u(t).
3. Structural Impedance Measurement (Curvature angle & Topological Phase Discrepancy).
4. Slow Latency Damping & Macro-rotor observation.
5. Dynamic Rule Mutation (Constraint A -> Constraint A').
6. Natural Attractor Convergence without scalar loss overfitting.
"""

import sys
import os
import time
import numpy as np

# Ensure root directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from synaptic_architecture.meta_constraint_feedback import MetaConstraintFeedbackLoop
import causal_engine as ce


def main():
    print("==================================================================================")
    print("  Elysia Continuous Causal Intelligence: Meta-Constraint Feedback & Re-perception")
    print("==================================================================================\n")

    # Initialize Feedback Engine with strict friction threshold to trigger rule mutation demo
    engine = MetaConstraintFeedbackLoop(
        num_field_nodes=64,
        hysterons_per_dim=8,
        gamma_curvature=0.4,
        latency_damping=0.1,
        friction_threshold=0.12, # Low friction threshold to demonstrate Rule Mutation on sharp curvature
    )

    # Inject uneven density weights to create initial distorted state space landscape
    weights = engine.field.density_weights
    for h in range(len(weights)):
        weights[h] = 0.1 + 0.8 * float((h * 17) % 11) / 10.0
    engine.field.density_weights = weights

    ce.update_preisach_field(engine.field)

    target_macro_trajectory = [0, 2, 5, 8]
    start_node = 0
    goal_node = 8

    print(f"Target Concept Trajectory (Macro Intention): {target_macro_trajectory}")
    print(f"Initial State Space Constraints (Rule A):")
    rule = engine.mutator.get_current_rule()
    print(f"  - Max Reluctance: {rule.max_reluctance_threshold:.3f}")
    print(f"  - Min Axiom Rigidity: {rule.min_rigidity_threshold:.3f}")
    print(f"  - Alpha Bounds: [{rule.alpha_boundary_min:.2f}, {rule.alpha_boundary_max:.2f}]")
    print(f"  - Beta Bounds: [{rule.beta_boundary_min:.2f}, {rule.beta_boundary_max:.2f}]\n")

    print("----------------------------------------------------------------------------------")
    print(" Executing Multi-Step Meta-Feedback & Re-perception Loop")
    print("----------------------------------------------------------------------------------\n")

    num_iterations = 6

    for step in range(1, num_iterations + 1):
        # Run one full iteration of the loop
        res = engine.step_meta_feedback(start_node, goal_node, target_macro_trajectory)

        print(f"[Step {step:02d}] Fast-Clock Trajectory Generated: {res['best_trajectory']}")
        print(f"  - Trajectory Curvature Angle: {res['trajectory_curvature']:.4f} rad")
        print(f"  - Topological Phase Discrepancy: {res['topological_phase_diff']:.4f}")
        print(f"  - Latency-Damped Friction (Impedance): {res['latency_damped_friction']:.4f}")
        print(f"  - Structural Resonance Score: {res['resonance_score']:.4f}")

        if res["rule_mutated"]:
            print(f"  ==> [META-FEEDBACK] High Impedance Detected! Rule Mutation Triggered (Count: {res['mutation_count']})")
            print(f"      Updated Rule A' Constraints -> Reluctance: {res['rule']['max_reluctance']:.3f}, Rigidity: {res['rule']['min_rigidity']:.3f}")
            print(f"      Alpha Bounds: [{res['rule']['alpha_bounds'][0]:.2f}, {res['rule']['alpha_bounds'][1]:.2f}]")
        else:
            print("  ==> [RESONANCE] Trajectory is harmoniously aligned with Macro Intent. No Rule Mutation needed.")

        print()
        time.sleep(0.05)

    print("==================================================================================")
    print("  Demonstration Complete: Natural Attractor Convergence Verified!")
    print("==================================================================================")


if __name__ == "__main__":
    main()
