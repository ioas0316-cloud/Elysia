"""
Unit tests for core/topology/semantic_reactive_engine.py
"""

import json
import pytest
from pydantic import ValidationError

from core.topology.semantic_reactive_engine import (
    SemanticState,
    Context,
    ContextFeedback,
    Axiom,
    TransitionRule,
    AxiomSchema,
    TransitionRuleSchema,
    AxiomRegistryConfig,
    DynamicAxiomRegistry,
    ResponsibilityOperator,
    TrustOperator,
    OrderOperator,
    LawOperator,
    CompassionOperator,
    SynthesizerOperator,
    LinguisticNode,
    LinguisticDependencyGraph,
)


def test_semantic_state_copy():
    state = SemanticState(identity="Agent1", qualities={"A", "B"}, relational_bindings={"x": "1"})
    copied = state.copy()
    assert copied.identity == "Agent1"
    assert copied.qualities == {"A", "B"}
    assert copied.relational_bindings == {"x": "1"}

    # Mutate copied and ensure original is unaffected
    copied.qualities.add("C")
    copied.relational_bindings["y"] = "2"
    assert "C" not in state.qualities
    assert "y" not in state.relational_bindings


def test_context_feedback_is_empty():
    fb = ContextFeedback()
    assert fb.is_empty() is True

    fb.events_to_add.add("EVENT_1")
    assert fb.is_empty() is False


def test_pydantic_axiom_schema_validation():
    # Valid config
    valid_data = {
        "axioms": [
            {"name": "RIGID_LAW", "priority": 1, "is_active": True},
            {"name": "RESTORATIVE_GRACE", "priority": 2},
        ],
        "transition_rules": [
            {"source": "RIGID_LAW", "trigger": "PARADOX_GRIDLOCK", "target": "RESTORATIVE_GRACE"}
        ],
    }
    cfg = AxiomRegistryConfig.model_validate(valid_data)
    assert len(cfg.axioms) == 2
    assert len(cfg.transition_rules) == 1

    # Invalid: duplicate axiom names
    dup_data = {
        "axioms": [
            {"name": "RIGID_LAW", "priority": 1},
            {"name": "RIGID_LAW", "priority": 2},
        ]
    }
    with pytest.raises(ValidationError):
        AxiomRegistryConfig.model_validate(dup_data)

    # Invalid: undefined axiom in transition rules
    invalid_rule_data = {
        "axioms": [{"name": "RIGID_LAW", "priority": 1}],
        "transition_rules": [{"source": "RIGID_LAW", "trigger": "X", "target": "UNKNOWN"}],
    }
    with pytest.raises(ValidationError):
        AxiomRegistryConfig.model_validate(invalid_rule_data)


def test_dynamic_axiom_registry():
    json_cfg = json.dumps({
        "axioms": [
            {"name": "AX1", "priority": 1, "is_active": True},
            {"name": "AX2", "priority": 2},
        ],
        "transition_rules": [
            {"source": "AX1", "trigger": "TRIG_1", "target": "AX2"}
        ]
    })
    registry = DynamicAxiomRegistry.load_from_json(json_cfg)
    assert registry.active_axiom is not None
    assert registry.active_axiom.name == "AX1"

    shifted = registry.trigger_phase_transition("TRIG_1")
    assert shifted is True
    assert registry.active_axiom.name == "AX2"


def test_reactive_cascade_and_topological_eval():
    subject = SemanticState("Agent_Alpha")
    graph = LinguisticDependencyGraph(base_subject=subject)

    graph.add_node(LinguisticNode("responsibility", ResponsibilityOperator(), {"subject": "subject"}))
    graph.add_node(LinguisticNode("trust", TrustOperator(), {"responsibility": "responsibility"}))
    graph.add_node(LinguisticNode("order", OrderOperator(), {"trust": "trust"}))

    # Scenario 1: Awareness, Choice, Crisis
    res = graph.propagate_until_equilibrium({"AWARENESS", "FREE_CHOICE", "CRISIS"})
    assert res["status"] == "EQUILIBRIUM_REACHED"
    states = res["states"]

    assert "ACCOUNTABLE" in states["responsibility"].qualities
    assert states["responsibility"].relational_bindings.get("CONSEQUENCE") == "SELF_SACRIFICE"

    assert "ABSOLUTE_ALIGNMENT" in states["trust"].qualities
    assert states["trust"].relational_bindings.get("RELATION") == "UNBREAKABLE_BOND"

    assert "HARMONIC_ORDER" in states["order"].qualities
    assert states["order"].relational_bindings.get("ENTROPY") == "MINIMIZED"


def test_recursive_feedback_equilibrium():
    subject = SemanticState("Agent_Beta")
    graph = LinguisticDependencyGraph(base_subject=subject)

    graph.add_node(LinguisticNode("responsibility", ResponsibilityOperator(), {"subject": "subject"}))
    graph.add_node(LinguisticNode("trust", TrustOperator(), {"responsibility": "responsibility"}))
    graph.add_node(LinguisticNode("order", OrderOperator(), {"trust": "trust"}))

    # Inject NEGLIGENCE -> Triggers CHAOS -> Feedback triggers SYSTEM_RESET & removes NEGLIGENCE -> Re-evaluates to STABILIZING
    res = graph.propagate_until_equilibrium({"NEGLIGENCE"})
    assert res["status"] == "EQUILIBRIUM_REACHED"
    assert res["iterations"] == 2

    states = res["states"]
    assert "RECOVERING" in states["responsibility"].qualities
    assert "RESTRICTED_TRUST" in states["trust"].qualities
    assert "STABILIZING" in states["order"].qualities


def test_dialectic_axiom_phase_transition():
    reg_cfg = {
        "axioms": [
            {"name": "RIGID_LAW", "priority": 1, "is_active": True},
            {"name": "RESTORATIVE_GRACE", "priority": 2},
        ],
        "transition_rules": [
            {"source": "RIGID_LAW", "trigger": "PARADOX_GRIDLOCK", "target": "RESTORATIVE_GRACE"}
        ]
    }
    registry = DynamicAxiomRegistry.load_from_config_dict(reg_cfg)
    graph = LinguisticDependencyGraph(axiom_registry=registry)

    graph.add_node(LinguisticNode("law", LawOperator(), {}))
    graph.add_node(LinguisticNode("compassion", CompassionOperator(), {}))
    graph.add_node(LinguisticNode("synthesis", SynthesizerOperator(), {"law": "law", "compassion": "compassion"}))

    # Inject SURVIVAL_CRIME -> Paradox Gridlock under RIGID_LAW -> Phase shift to RESTORATIVE_GRACE -> Harmonized synthesis
    res = graph.propagate_until_equilibrium({"SURVIVAL_CRIME"})
    assert res["status"] == "EQUILIBRIUM_REACHED"
    assert registry.active_axiom.name == "RESTORATIVE_GRACE"

    states = res["states"]
    assert "CONTEXTUAL_MERCY" in states["law"].qualities
    assert states["synthesis"].relational_bindings.get("STATUS") == "HARMONIZED_REDEEM"
