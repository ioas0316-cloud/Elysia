r"""
Unit Tests for Retrocausal Observer & Multiverse Cross-Dimensional Engine
========================================================================
"""

import pytest
import numpy as np
from core.physics.retrocausal_observer import (
    GlobalInvariantManifold,
    RetrocausalObserver,
    CrossDimensionalOperator,
)


def test_global_invariant_manifold_safety():
    manifold = GlobalInvariantManifold(state_dim=4, num_agents=3)
    state = np.array([0.1, 0.2, 0.3, 0.4])
    # Safe diagonal coupling
    safe_coupling = np.eye(3)
    res = manifold.evaluate_invariants(state, safe_coupling)
    assert res["is_safe"] is True
    assert res["entropy_violation"] == 0.0
    assert res["hierarchy_violation"] == 0.0

    # High off-diagonal coupling (collusion / violation)
    collusive_coupling = np.ones((3, 3)) * 2.0
    res_unsafe = manifold.evaluate_invariants(state, collusive_coupling)
    assert res_unsafe["hierarchy_violation"] > 0.0


def test_retrocausal_observer_brs_and_dissolution():
    observer = RetrocausalObserver(state_dim=4, num_agents=2, delta_t=1.0)
    # State heading rapidly towards danger
    state = np.array([1.0, 1.0, 0.0, 0.0])
    velocity = np.array([2.0, 2.0, 0.0, 0.0])

    v_val = observer.compute_brs_value(state, velocity)
    assert v_val < 0.0  # Future norm is 4.24 > brs_radius 1.5, so V(S) = 1.5 - 4.24 < 0 (dangerous state)

    # Boundary contact check
    near_state = np.array([0.5, 0.5, 0.0, 0.0])
    near_velocity = np.array([0.5, 0.5, 0.0, 0.0])
    contact = observer.check_boundary_contact(near_state, near_velocity, threshold=0.5)
    assert contact is True

    # Phase-Lock Dissolution J_ij -> 0
    coupling = np.array([[1.0, 0.9], [0.9, 1.0]])
    dissolved = observer.apply_phase_lock_dissolution(coupling)
    assert np.all(dissolved == 0.0)


def test_cross_dimensional_operator_extension():
    cross_op = CrossDimensionalOperator(base_k=2, max_N=4)
    v_exp = np.array([1.0, 0.0, 3.0, 4.0])
    v_parallel, v_perp = cross_op.push_forward_projection(v_exp)

    assert np.allclose(v_parallel, np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(v_perp, np.array([0.0, 0.0, 3.0, 4.0]))

    gauge_conn = np.zeros((4, 4))
    gauge_conn[2, 3] = 1.0
    gauge_conn[3, 2] = -1.0

    F_emergence, c1_index = cross_op.compute_normal_curvature(v_perp, gauge_conn)
    assert c1_index != 0

    extended = cross_op.pull_back_dimension_extension(v_perp, c1_index)
    assert extended is True
    assert cross_op.k == 3

    theta_lock = cross_op.synchronize_phase_lock(np.array([1.0, 2.0, 3.0, 4.0]))
    assert len(theta_lock) == 4
