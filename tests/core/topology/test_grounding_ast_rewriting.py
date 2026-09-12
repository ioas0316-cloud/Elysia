"""
Unit tests for Grounding Ontology Engine and AST Self-Rewriting Engine.
"""

import pytest
from core.topology.grounding_ontology import GroundingAxiom, GroundingOntologyEngine
from core.topology.ast_rewriting import SafeASTRewritingEngine, SelfRewritingNodeEngine


def test_grounding_ontology_validation():
    ontology = GroundingOntologyEngine()
    thermal_safety_axiom = GroundingAxiom(
        axiom_id="AXIOM_THERMAL_01",
        description="열 위험 상태의 물리적 모순 방지 및 보호 구속 명시",
        forbidden_pairs=[("THERMAL_HAZARD", "FROST_CRYSTAL")],
        required_bindings={"SAFETY_CONTAINMENT": "ACTIVE"},
    )
    ontology.register_axiom(thermal_safety_axiom)

    invalid_qualities = {"THERMAL_HAZARD", "FROST_CRYSTAL"}
    invalid_bindings = {"SAFETY_CONTAINMENT": "INACTIVE"}

    is_valid, errors = ontology.validate_semantic_state(invalid_qualities, invalid_bindings)
    assert not is_valid
    assert len(errors) == 2


def test_self_rewriting_node_engine():
    engine = SelfRewritingNodeEngine()

    def initial_operator(state):
        return {"qualities": set(state.get("qualities", [])) | {"NORMAL"}, "bindings": state.get("bindings", {})}

    engine.register_operator("Causal_Node_A", initial_operator)

    res1 = engine.execute_node("Causal_Node_A", {"qualities": ["RAW_DATA"]})
    assert "NORMAL" in res1["qualities"]

    engine.evaluate_and_rewrite("Causal_Node_A", confidence_score=0.32)

    res2 = engine.execute_node("Causal_Node_A", {"qualities": ["RAW_DATA"]})
    assert "UNCERTAINTY_ISOLATED" in res2["qualities"]
    assert res2["bindings"]["EXECUTION_MODE"] == "SAFE_FALLBACK"


def test_safe_ast_rewriting_engine():
    engine = SafeASTRewritingEngine()
    original_source = """
def sample_operator(state):
    qualities = set(state.get('qualities', []))
    qualities.add('FROST_CRYSTAL')

    bindings = {
        'SAFETY_CONTAINMENT': 'INACTIVE'
    }
    return {'qualities': qualities, 'bindings': bindings}
"""
    engine.register_from_source("Causal_Node_AST", original_source)
    res_before = engine.execute("Causal_Node_AST", {"qualities": {"THERMAL_HAZARD"}})
    assert res_before["bindings"]["SAFETY_CONTAINMENT"] == "INACTIVE"

    engine.rewrite_and_compile(
        node_id="Causal_Node_AST",
        original_source=original_source,
        remove_q="FROST_CRYSTAL",
        req_k="SAFETY_CONTAINMENT",
        req_v="ACTIVE_BY_AST",
    )

    res_after = engine.execute("Causal_Node_AST", {"qualities": {"THERMAL_HAZARD"}})
    assert res_after["bindings"]["SAFETY_CONTAINMENT"] == "ACTIVE_BY_AST"
