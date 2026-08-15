"""
Unit tests for EquivalenceTuningEngine (core/topology/equivalence_tuning_engine.py)
"""

import numpy as np
import pytest

from core.topology.causal_structure import InformationTopology, CausalNumber, CausalSymbol, TopologyLink
from core.topology.topological_comparer import TopologicalComparer
from core.topology.causal_discernment_engine import CausalDiscernmentEngine
from core.topology.equivalence_tuning_engine import (
    IntentAnchor,
    FunctionalMap,
    EquivalenceVerifier,
    EquivalenceTuningEngine,
    EquivalenceVerificationResult
)


def test_intent_anchor_creation():
    target_topo = InformationTopology("TargetTopo")
    target_metric = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    intent = IntentAnchor(
        intent_id="intent_01",
        target_topology=target_topo,
        target_metric=target_metric,
        tolerance=0.05
    )

    assert intent.intent_id == "intent_01"
    assert intent.target_topology.name == "TargetTopo"
    assert np.array_equal(intent.target_metric, target_metric)
    assert intent.tolerance == 0.05


def test_functional_map_transformation_and_tuning():
    fmap = FunctionalMap(input_dim=4, output_dim=4, learning_rate=0.2)
    x = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Transform initial
    y_init = fmap.transform(x)
    assert len(y_init) == 4

    # Apply error backprop tuning
    error_delta = np.array([0.5, -0.5, 0.0, 0.0], dtype=np.float32)
    fmap.tune_backprop(x, error_delta)

    # Transform post tuning
    y_post = fmap.transform(x)
    # y_post should move closer to target (original y - lr * error)
    assert y_post[0] < y_init[0]
    assert y_post[1] > y_init[1]


def test_equivalence_verifier():
    verifier = EquivalenceVerifier()
    target_topo = InformationTopology("TargetTopo")
    intent = IntentAnchor(
        intent_id="intent_01",
        target_topology=target_topo,
        target_metric=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        tolerance=0.1
    )

    # Equal case
    prod_metric = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    res = verifier.verify(prod_metric, target_topo, intent)
    assert res.is_equivalent is True
    assert res.phase_disparity == 0.0
    assert res.equivalence_degree >= 0.99

    # Disparate case
    prod_disparate = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)
    res_disp = verifier.verify(prod_disparate, target_topo, intent)
    assert res_disp.is_equivalent is False
    assert res_disp.phase_disparity > 0.1


def test_equivalence_tuning_engine_closed_loop_convergence():
    fmap = FunctionalMap(input_dim=4, output_dim=4, learning_rate=0.1)
    verifier = EquivalenceVerifier()
    discernment_engine = CausalDiscernmentEngine()

    engine = EquivalenceTuningEngine(
        functional_map=fmap,
        verifier=verifier,
        discernment_engine=discernment_engine,
        max_tuning_iterations=30
    )

    target_topo = InformationTopology("TargetTopo")
    target_metric = np.array([2.5, -1.0, 0.5, 3.0], dtype=np.float32)

    intent = IntentAnchor(
        intent_id="intent_closed_loop",
        target_topology=target_topo,
        target_metric=target_metric,
        tolerance=0.08
    )

    input_stimulus = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    final_produced, final_verification, history = engine.run_ouroboros_loop(
        input_stimulus=input_stimulus,
        intent=intent
    )

    # Check loop progress and convergence
    assert len(history) > 0
    assert final_verification.is_equivalent is True
    assert final_verification.phase_disparity <= intent.tolerance
    assert history[-1].equivalence_degree >= history[0].equivalence_degree
