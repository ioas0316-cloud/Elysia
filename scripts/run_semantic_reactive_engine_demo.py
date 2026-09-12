"""
Elysia Core Architecture: Semantic Reactive Engine Demonstration Script

This script demonstrates the 4 core capabilities of the Semantic Reactive Engine:
1. Reactive Cascade (CAD / Excel parametric dependency propagation)
2. Recursive Context Feedback Loop (Homeostatic equilibrium convergence)
3. Dialectic Paradox Resolution (Axiom Phase Transition)
4. Dynamic JSON Schema Loading & Integrity Validation (Pydantic v2)
"""

import json
from core.topology.semantic_reactive_engine import (
    SemanticState,
    ResponsibilityOperator,
    TrustOperator,
    OrderOperator,
    LawOperator,
    CompassionOperator,
    SynthesizerOperator,
    LinguisticNode,
    LinguisticDependencyGraph,
    DynamicAxiomRegistry,
)


def run_demo_1_reactive_cascade():
    print("\n" + "=" * 70)
    print("DEMO 1: Reactive Cascade (반응형 연쇄 전파)")
    print("=" * 70)

    subject = SemanticState("Agent_Alpha")
    graph = LinguisticDependencyGraph(base_subject=subject)

    graph.add_node(LinguisticNode("responsibility", ResponsibilityOperator(), {"subject": "subject"}))
    graph.add_node(LinguisticNode("trust", TrustOperator(), {"responsibility": "responsibility"}))
    graph.add_node(LinguisticNode("order", OrderOperator(), {"trust": "trust"}))

    events = {"AWARENESS", "FREE_CHOICE", "CRISIS"}
    print(f"🚀 [최상위 맥락 주입]: {events}")

    res = graph.propagate_until_equilibrium(events)
    print(f"📊 [결과 수렴 상태]: {res['status']} (Iteration {res['iterations']})")
    for name, state in res["states"].items():
        print(f"  └─ [{name}]: 특성={state.qualities} | 관계={state.relational_bindings}")


def run_demo_2_feedback_equilibrium():
    print("\n" + "=" * 70)
    print("DEMO 2: Recursive Feedback Loop (역피드백 자율 수렴)")
    print("=" * 70)

    subject = SemanticState("Agent_Beta")
    graph = LinguisticDependencyGraph(base_subject=subject)

    graph.add_node(LinguisticNode("responsibility", ResponsibilityOperator(), {"subject": "subject"}))
    graph.add_node(LinguisticNode("trust", TrustOperator(), {"responsibility": "responsibility"}))
    graph.add_node(LinguisticNode("order", OrderOperator(), {"trust": "trust"}))

    events = {"NEGLIGENCE"}
    print(f"⚡ [최상위 마찰 맥락 주입]: {events}")

    res = graph.propagate_until_equilibrium(events)
    print(f"📊 [결과 수렴 상태]: {res['status']} ({res['iterations']}회차 만에 항성성 수렴)")
    for log_entry in res["log"]:
        print(f"  [Iter {log_entry['iteration']}] Context={log_entry['context']}")
        for name, state in log_entry["states"].items():
            print(f"    └─ [{name}]: {state.qualities} | {state.relational_bindings}")


def run_demo_3_axiom_phase_transition():
    print("\n" + "=" * 70)
    print("DEMO 3: Dialectic Paradox Resolution via Axiom Shift (변증법적 공리 상전이)")
    print("=" * 70)

    json_config = json.dumps({
        "axioms": [
            {"name": "RIGID_LAW", "priority": 1, "is_active": True, "metadata": {"desc": "엄격한 원칙주의"}},
            {"name": "RESTORATIVE_GRACE", "priority": 2, "metadata": {"desc": "회복적 긍휼 공리"}},
        ],
        "transition_rules": [
            {"source": "RIGID_LAW", "trigger": "PARADOX_GRIDLOCK", "target": "RESTORATIVE_GRACE"}
        ]
    })

    registry = DynamicAxiomRegistry.load_from_json(json_config)
    graph = LinguisticDependencyGraph(axiom_registry=registry)

    graph.add_node(LinguisticNode("law", LawOperator(), {}))
    graph.add_node(LinguisticNode("compassion", CompassionOperator(), {}))
    graph.add_node(LinguisticNode("synthesis", SynthesizerOperator(), {"law": "law", "compassion": "compassion"}))

    events = {"SURVIVAL_CRIME"}
    print(f"🏛️ [초기 가동 공리]: {registry.active_axiom.name}")
    print(f"💥 [충돌 맥락 주입]: {events}")

    res = graph.propagate_until_equilibrium(events)
    print(f"\n✨ [최종 평형 도달]: {res['status']} | 활성 공리: {registry.active_axiom.name}")
    for name, state in res["states"].items():
        print(f"  └─ [{name}]: {state.qualities} | {state.relational_bindings}")


def main():
    print("======================================================================")
    print("      Elysia Core: Semantic Reactive Engine Verification Demo        ")
    print("======================================================================")

    run_demo_1_reactive_cascade()
    run_demo_2_feedback_equilibrium()
    run_demo_3_axiom_phase_transition()

    print("\n" + "=" * 70)
    print("✅ All Semantic Reactive Engine Demos Successfully Executed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
