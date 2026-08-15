"""
Tests for Causal Assembly Framework
===================================
Tests FixedEdgeDependencyMap, StaticStateMatrix, RetroactiveIntentAnchor,
and CausalAssemblyEngine.
"""

import numpy as np
import pytest
from core.consciousness.causal_assembly_engine import (
    FixedEdgeDependencyMap,
    StaticStateMatrix,
    RetroactiveIntentAnchor,
    CausalAssemblyEngine
)
from core.utils.math_utils import Quaternion


def test_fixed_edge_dependency_map():
    dep_map = FixedEdgeDependencyMap()

    # Define simple fragments
    def data_source():
        return {"raw_val": 42}

    def transformer(data_source=None):
        val = data_source["raw_val"] if data_source else 0
        return {"transformed_val": val * 2}

    def aggregator(transformer=None):
        t_val = transformer["transformed_val"] if transformer else 0
        return f"Result: {t_val}"

    dep_map.register_fragment("data_source", data_source)
    dep_map.register_fragment("transformer", transformer)
    dep_map.register_fragment("aggregator", aggregator)

    dep_map.add_causal_edge("data_source", "transformer")
    dep_map.add_causal_edge("transformer", "aggregator")

    order = dep_map.get_execution_order()
    assert order == ["data_source", "transformer", "aggregator"]

    results = dep_map.propagate({})
    assert results["data_source"] == {"raw_val": 42}
    assert results["transformer"] == {"transformed_val": 84}
    assert results["aggregator"] == "Result: 84"


def test_static_state_matrix():
    matrix = StaticStateMatrix(key_dim=4)

    # Register pre-validated pathways
    vec1 = np.array([1.0, -1.0, 1.0, 1.0])
    vec2 = np.array([-1.0, 1.0, -1.0, -1.0])

    matrix.register_pathway("1_-1_1_1", vec1, "Outcome_A")
    matrix.register_pathway("-1_1_-1_-1", vec2, "Outcome_B")

    # O(1) direct lookup test
    input_v1 = np.array([2.5, -0.1, 10.0, 0.5])
    outcome, friction = matrix.evaluate(input_v1)
    assert outcome == "Outcome_A"
    assert friction == 0.0

    # Near lookup / matrix projection fallback test
    input_v2 = np.array([-0.8, 0.9, -1.1, -0.5])
    outcome2, friction2 = matrix.evaluate(input_v2)
    assert outcome2 == "Outcome_B"
    assert friction2 < 0.2


def test_retroactive_intent_anchor():
    anchor = RetroactiveIntentAnchor()

    outcome_a = {"status": "SUCCESS", "metric": 100}
    q_outcome = anchor.derive_intent_from_outcome(outcome_a)
    assert isinstance(q_outcome, Quaternion)

    # Set anchor to this derived outcome
    anchor.intent_anchor = q_outcome

    # Measuring divergence against identical outcome
    divergence = anchor.measure_phase_divergence(q_outcome)
    assert divergence < 1e-5

    # Measuring divergence against different outcome
    q_different = Quaternion(0.0, 1.0, 0.0, 0.0).normalize()
    div_different = anchor.measure_phase_divergence(q_different)
    assert div_different > 0.1

    # Test variable feedback generation
    vars_in = np.array([1.0, 2.0, 3.0, 4.0])
    adjusted_vars, pd = anchor.generate_variable_feedback(vars_in, q_different)
    assert pd > 0.0
    assert adjusted_vars.shape == vars_in.shape
    assert not np.array_equal(vars_in, adjusted_vars)


def test_causal_assembly_engine():
    engine = CausalAssemblyEngine(key_dim=4)

    # Register fragments
    def fetch_api():
        return {"response": 200, "data": "payload"}

    def process_payload(fetch_api=None):
        return f"processed_{fetch_api['data']}"

    fragments = {
        "fetch_api": fetch_api,
        "process_payload": process_payload
    }
    edges = [("fetch_api", "process_payload")]

    static_pathways = [
        {"key": "1_1_1_1", "vector": [1.0, 1.0, 1.0, 1.0], "outcome": "STABLE_ROUTE"}
    ]

    engine.assemble_fragments(fragments, edges, static_pathways)

    # Run cycle
    var_inputs = np.array([1.0, 1.0, 1.0, 1.0])
    cycle_res = engine.run_causal_cycle({}, var_inputs)

    assert cycle_res["propagated_results"]["fetch_api"] == {"response": 200, "data": "payload"}
    assert cycle_res["propagated_results"]["process_payload"] == "processed_payload"
    assert cycle_res["matrix_outcome"] == "STABLE_ROUTE"
    assert "phase_divergence" in cycle_res
    assert "adjusted_variables" in cycle_res
