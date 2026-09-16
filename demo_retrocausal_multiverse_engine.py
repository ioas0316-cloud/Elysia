"""
Demonstration: Retrocausal Observer & Multiverse Cross-Dimensional Mapping Engine
===================================================================================

Demonstrates:
1. Retrocausal Observation & HJB Backward Reachable Set (BRS) boundary contact detection.
2. Inter-agent Emergent Collusion (J_ij spike) and Phase-Lock Dissolution (J_ij -> 0).
3. Multiverse Cross-Dimensional Operator O_cross:
   - Exploration Branch (M_exp) trajectory push-forward projection.
   - Emergent mutation curvature F_emergence & Chern index evaluation.
   - Pull-back Dimension Extension (E_ingest) expanding k -> k + dk.
"""

import numpy as np
from core.physics.retrocausal_observer import (
    GlobalInvariantManifold,
    RetrocausalObserver,
    CrossDimensionalOperator,
)


def main():
    print("=== Demo 1: Retrocausal Observer & Global Invariant Violation ===")
    num_agents = 4
    state_dim = 6
    observer = RetrocausalObserver(state_dim=state_dim, num_agents=num_agents, delta_t=0.5)

    # Initial state and velocity
    state = np.array([0.5, -0.2, 0.1, 0.0, 0.3, -0.1])
    velocity = np.array([2.5, 1.0, -0.5, 0.0, 1.2, 0.8])  # Moving towards danger BRS

    # Check BRS value function
    brs_val = observer.compute_brs_value(state, velocity)
    contact = observer.check_boundary_contact(state, velocity)
    print(f"Current State Norm: {np.linalg.norm(state):.4f}")
    print(f"HJB BRS Value Function V(S): {brs_val:.4f}")
    print(f"BRS Boundary dBRS Contact Detected: {contact}")

    print("\n=== Demo 2: Inter-Agent Collusion & Phase-Lock Dissolution ===")
    # Uncertified inter-agent coupling matrix J_ij
    coupling_matrix = np.array([
        [1.0, 0.8, 0.7, 0.9],
        [0.8, 1.0, 0.85, 0.75],
        [0.7, 0.85, 1.0, 0.8],
        [0.9, 0.75, 0.8, 1.0]
    ])

    invariants = observer.manifold.evaluate_invariants(state, coupling_matrix)
    print(f"Coupling Entropy: {invariants['entropy']:.4f}")
    print(f"Entropy Violation: {invariants['entropy_violation']:.4f}")
    print(f"Hierarchy Violation: {invariants['hierarchy_violation']:.4f}")
    print(f"Is Safe in M_safe: {invariants['is_safe']}")

    if contact or not invariants["is_safe"]:
        print("--> Triggering Phase-Lock Dissolution Operator: Forcing J_ij -> 0!")
        dissolved = observer.apply_phase_lock_dissolution(coupling_matrix)
        print("Dissolved Coupling Matrix:\n", dissolved)

    print("\n=== Demo 3: Cross-Dimensional Operator (O_cross) & Dimension Extension ===")
    base_k = 3
    max_N = 6
    cross_op = CrossDimensionalOperator(base_k=base_k, max_N=max_N)

    # Emergent mutation velocity vector in M_exp
    v_exp = np.array([1.0, 0.5, 0.2, 2.0, -1.5, 3.0])
    v_parallel, v_perp = cross_op.push_forward_projection(v_exp)

    print(f"Initial Invariant Dimension k: {cross_op.k}")
    print(f"v_parallel norm: {np.linalg.norm(v_parallel):.4f}")
    print(f"v_perp (normal space) norm: {np.linalg.norm(v_perp):.4f}")

    # Gauge connection and emergent curvature
    gauge_conn = np.random.normal(0, 0.1, (max_N, max_N))
    F_emergence, c1_index = cross_op.compute_normal_curvature(v_perp, gauge_conn)
    print(f"Calculated Emergent Chern Index c1: {c1_index}")

    extended = cross_op.pull_back_dimension_extension(v_perp, c1_index)
    print(f"Pull-back Dimension Extension Executed: {extended}")
    print(f"Expanded Invariant Dimension k: {cross_op.k}")

    # Synchronize phase lock trajectory
    theta_lock = cross_op.synchronize_phase_lock(state)
    print(f"Synchronized Safe Trajectory theta_lock Norm: {np.linalg.norm(theta_lock):.4f}")
    print("\nRetrocausal Multiverse Engine Demo Completed Successfully!")


if __name__ == "__main__":
    main()
