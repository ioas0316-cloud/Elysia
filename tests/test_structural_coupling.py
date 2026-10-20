"""
Unit tests for ELYSIA Structural Coupling & Discrepancy Tracking Engine v0.2
Verifies the 4 Steps:
1. Existing relation inference (1 -> 2 -> 3) and proof protocol logging.
2. Discrepancy classification (Noise vs. Structural Discrepancy).
3. The 4 Structural Reactions: MAINTAIN, MODIFY, BRANCH (Mitosis), SUSPEND_JUDGMENT.
4. Re-observation & Future prediction evolution across parallel topological layers.
"""

import pytest
from core.topology.structural_coupling_engine import (
    StructuralRelationalGraph,
    StructuralCouplingEngine,
    DiscrepancyAnalysis
)


def test_step1_relation_inference_and_proof_protocol():
    engine = StructuralCouplingEngine()
    engine.record_initial_relation(["1", "2", "3"])

    graph = engine.primary_graph
    assert ("1", "2") in graph.edges
    assert ("2", "3") in graph.edges

    # Predict next state from state "1"
    predicted = graph.predict_next_state("1")
    assert predicted == "2"

    # Verify proof protocol recorded steps
    steps = graph.proof_protocol.steps
    assert len(steps) >= 3  # Two edge additions + one inference step
    rules = [s.rule_applied for s in steps]
    assert "ADD_RELATIONAL_EDGE" in rules
    assert "INFER_NEXT_STATE" in rules


def test_step2_and_3_reaction_maintain_on_noise():
    engine = StructuralCouplingEngine(noise_threshold=0.15)
    engine.record_initial_relation(["1", "2", "3"])

    # Observation matches 1 -> 2 with minor value noise (0.05)
    res = engine.process_observation(
        current_state="1",
        observed_next_state="2",
        observed_value_delta=0.05
    )

    assert res["reaction"] == "MAINTAIN"
    assert res["analysis"].discrepancy_type == "NOISE"
    # Verify graph edge weight remains 1.0
    assert engine.primary_graph.edges[("1", "2")].weight == 1.0


def test_step2_and_3_reaction_modify_on_structural_discrepancy():
    engine = StructuralCouplingEngine(noise_threshold=0.15)
    engine.record_initial_relation(["1", "2", "3"])

    # Unexpected observation 2 -> 4 (expected 2 -> 3)
    res = engine.process_observation(
        current_state="2",
        observed_next_state="4",
        observed_value_delta=1.0
    )

    assert res["reaction"] == "MODIFY"
    assert res["analysis"].discrepancy_type == "STRUCTURAL_DISCREPANCY"
    # Verify edge 2 -> 4 was added with higher weight and 2 -> 3 weight was decayed
    assert ("2", "4") in engine.primary_graph.edges
    assert engine.primary_graph.edges[("2", "4")].weight == 1.2
    assert engine.primary_graph.edges[("2", "3")].weight == 0.5


def test_step2_and_3_reaction_branch_mitosis_on_context_conflict():
    engine = StructuralCouplingEngine()
    engine.record_initial_relation(["1", "2", "3"])

    # Observation under context "high_pressure" yields 2 -> 5 instead of 2 -> 3
    res = engine.process_observation(
        current_state="2",
        observed_next_state="5",
        context_condition="high_pressure"
    )

    assert res["reaction"] == "BRANCH"
    assert res["analysis"].discrepancy_type == "MUTUALLY_EXCLUSIVE_CONFLICT"
    assert "context_high_pressure" in engine.active_layers

    # Check primary graph retains 2 -> 3
    assert engine.predict_future_under_layer("2", "main_layer") == "3"
    # Check branched graph predicts 2 -> 5
    assert engine.predict_future_under_layer("2", "context_high_pressure") == "5"


def test_step2_and_3_reaction_suspend_judgment_on_ambiguity():
    engine = StructuralCouplingEngine()
    engine.record_initial_relation(["1", "2", "3"])

    # Ambiguous observation
    res = engine.process_observation(
        current_state="2",
        observed_next_state="UNKNOWN_X",
        is_ambiguous_data=True
    )

    assert res["reaction"] == "SUSPEND_JUDGMENT"
    assert res["analysis"].discrepancy_type == "AMBIGUOUS_INSUFFICIENT"
    # Verify no new edge added
    assert ("2", "UNKNOWN_X") not in engine.primary_graph.edges


def test_step4_reobservation_and_future_prediction_evolution():
    engine = StructuralCouplingEngine()
    engine.record_initial_relation(["1", "2", "3"])

    # Initial prediction from 2 is 3
    p1 = engine.primary_graph.predict_next_state("2")
    assert p1 == "3"

    # Process structural change observation 2 -> 4
    engine.process_observation("2", "4", observed_value_delta=1.0)

    # Re-predict from 2 after structural reaction
    p2 = engine.primary_graph.predict_next_state("2")
    assert p2 == "4"  # Prediction actively changed due to updated relational topology
