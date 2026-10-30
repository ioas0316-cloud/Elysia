"""
Unit tests for SelfBoundaryTracker and RelationalKnowledgeGraph
in L2_cognitive_engine.
"""

import pytest
from L2_cognitive_engine.self_boundary_tracker import SelfBoundaryTracker, AttributionScore
from L2_cognitive_engine.relational_knowledge_graph import RelationalKnowledgeGraph


def test_self_boundary_tracker_attribution_normalization():
    score = AttributionScore(internal_origin=2.0, external_influence=2.0, ignorance_charge=1.0, criteria_revision=1.0)
    norm = score.normalize()

    assert norm.internal_origin == pytest.approx(2.0 / 6.0)
    assert norm.external_influence == pytest.approx(2.0 / 6.0)
    assert norm.ignorance_charge == pytest.approx(1.0 / 6.0)
    assert norm.criteria_revision == pytest.approx(1.0 / 6.0)


def test_self_boundary_tracker_evaluation_and_metacognition():
    tracker = SelfBoundaryTracker()

    state1 = tracker.evaluate_boundary(
        context_id="ctx_001",
        internal_reasoning="Derived response from internal prior $0_{value}$",
        external_stimulus="User asked a question about user experience",
        raw_attribution={
            "internal_origin": 0.5,
            "external_influence": 0.3,
            "ignorance_charge": 0.1,
            "criteria_revision": 0.1
        }
    )

    assert state1.context_id == "ctx_001"
    assert len(tracker.history) == 1

    # Evaluate second interaction with explicit unknown marker
    state2 = tracker.evaluate_boundary(
        context_id="ctx_002",
        internal_reasoning="Recognized unobserved boundary condition",
        external_stimulus="사용자가 모른다고 답변함 (unknown condition)",
        raw_attribution={
            "internal_origin": 0.2,
            "external_influence": 0.4,
            "ignorance_charge": 0.2,
            "criteria_revision": 0.2
        }
    )

    assert len(state2.unobserved_boundary) > 0
    assert len(state2.revised_criteria_notes) > 0

    summary = tracker.get_metacognitive_summary()
    assert summary["total_evaluations"] == 2
    assert summary["openness_index"] > 0.8  # Epistemic openness increased due to revision
    assert len(summary["recent_unobserved_boundaries"]) > 0


def test_relational_knowledge_graph_triadic_knowing():
    graph = RelationalKnowledgeGraph()

    triad = graph.add_triadic_knowing(
        object_id="world_object_01",
        object_label="External World Event",
        observer_id="elysia_self",
        observer_lens={"active_lens": "3_way_knowing"},
        interspace_properties={"friction_level": 0.25, "resonance": 0.8}
    )

    assert triad["object"].id == "world_object_01"
    assert triad["observer"].id == "elysia_self"
    assert len(triad["edges"]) == 2

    interspace_edges = graph.query_interspace("elysia_self", "world_object_01")
    assert len(interspace_edges) == 2


def test_relational_knowledge_graph_emotion_knowledge_network():
    graph = RelationalKnowledgeGraph()

    res = graph.register_emotion_knowledge_network(
        emotion_name="Sorrow",
        formation_conditions=["Loss of relational connection", "Unmet expectation"],
        causal_triggers=["Separation event"],
        consequent_behaviors=["Introspective withdrawal", "Seeking comfort"]
    )

    node = res["emotion_node"]
    assert node.label == "Sorrow"
    assert node.attributes["is_causal_knowledge"] is True
    assert node.attributes["is_biological_experience_asserted"] is False
    assert res["related_edges_count"] == 4
