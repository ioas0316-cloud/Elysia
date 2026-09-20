"""
Unit tests for Human Cognitive Phase Dynamics Module.
"""

import numpy as np
import pytest

from core.consciousness.human_cognitive_phase_dynamics import (
    MultiScalePhaseCouplingEngine,
    HebbianPhasePlasticity,
    MetaCognitiveCriticalityGovernor,
    HierarchicalAttractorNetwork
)


def test_multi_scale_phase_coupling_engine_step():
    num_nodes = 16
    engine = MultiScalePhaseCouplingEngine(num_nodes=num_nodes, dt=0.01)

    initial_slow = engine.slow_phase.copy()
    initial_fast = engine.fast_phase.copy()

    res = engine.step()

    assert "slow_phase" in res
    assert "fast_phase" in res
    assert "order_R_slow" in res
    assert "order_R_fast" in res
    assert "delta_phi_fast" in res

    assert res["slow_phase"].shape == (num_nodes,)
    assert res["fast_phase"].shape == (num_nodes,)
    assert 0.0 <= res["order_R_fast"] <= 1.0


def test_hebbian_phase_plasticity():
    num_nodes = 16
    plasticity = HebbianPhasePlasticity(num_nodes=num_nodes)

    metric = np.eye(num_nodes, dtype=np.float32) * 2.0
    slow_phase = np.zeros(num_nodes, dtype=np.float32)
    fast_phase = np.zeros(num_nodes, dtype=np.float32)

    updated_metric = plasticity.update_metric(metric, slow_phase, fast_phase)

    assert updated_metric.shape == (num_nodes, num_nodes)
    assert np.allclose(np.diag(updated_metric), 0.0)


def test_meta_cognitive_criticality_governor():
    num_nodes = 16
    engine = MultiScalePhaseCouplingEngine(num_nodes=num_nodes)
    governor = MetaCognitiveCriticalityGovernor()

    # Low error -> Rigid fixation adjustment
    res_rigid = governor.adapt_criticality(engine, delta_phi_fast=0.05, recalled_attractor_found=True)
    assert res_rigid["cognitive_state"] == "RIGID_FIXATION_PREJUDICE"

    # High error with no attractor -> Active learning adjustment
    res_learn = governor.adapt_criticality(engine, delta_phi_fast=0.80, recalled_attractor_found=False)
    assert res_learn["cognitive_state"] == "ACTIVE_NOVEL_CONCEPT_LEARNING"


def test_hierarchical_attractor_network():
    num_nodes = 16
    net = HierarchicalAttractorNetwork(num_nodes=num_nodes, resonance_threshold=0.50)

    fast_phase = np.zeros(num_nodes, dtype=np.float32)
    slow_phase = np.zeros(num_nodes, dtype=np.float32)

    attr_id_1 = net.store_1st_order_attractor("Test1", fast_phase)
    assert attr_id_1 == 0

    recalled = net.recall_1st_order(fast_phase)
    assert recalled is not None
    assert recalled["attractor"]["label"] == "Test1"

    meta_id = net.store_2nd_order_meta_attractor("Meta1", [0], slow_phase)
    assert meta_id == 0

    recalled_meta = net.recall_2nd_order(slow_phase)
    assert recalled_meta is not None
    assert recalled_meta["attractor"]["label"] == "Meta1"
