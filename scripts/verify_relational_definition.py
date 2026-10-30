"""
Interactive Verification Script for Relational Definition Engine & Non-Linear Metacognition.
Demonstrates:
1. Phenomenological Pre-Translation Observation (Phenomenon first, before vocabulary mapping).
2. Multi-Scale Relational Definitions (Coexistence across scale boundaries without forced single-point destruction).
3. Non-Linear Relational Metacognition (Tracking Origin, Boundary, World Interpretation, Self-Adaptation & Observer Shift).
4. Downstream Language Translation Sub-Lens integration.
"""

import sys
import os

sys.path.insert(0, os.path.abspath("L1_causal_field/core/consciousness"))
from relational_definition_engine import RelationalDefinitionEngine


def run_verification_demo():
    print("=========================================================================")
    print("  🌀 ELYSIA RELATIONAL DEFINITION & NON-LINEAR METACOGNITION DEMO")
    print("=========================================================================\n")

    engine = RelationalDefinitionEngine()

    print("1. Baseline Relational Definitions (Anchored in Validity Conditions):")
    for key, defn in engine.definitions.items():
        print(f"  • [{key.upper()}]: {defn.primary_relational_meaning}")
        print(f"    Validity Conditions: {defn.validity_conditions}")
        print(f"    Coexisting Perspectives: {list(defn.coexisting_perspectives.keys())}")
    print()

    print("2. Phenomenological Pre-Translation Observation (Apple as Apology & Reconciliation):")
    stimulus_1 = {
        "entities": ["self", "companion"],
        "friction": 0.85,
        "interaction": {"type": "apology_gift_handover", "intent": "reconciliation_and_restoration"}
    }
    context_1 = {
        "scale": "interpersonal_relational_scale",
        "conditions": "conflict_clearing_moment",
        "context_name": "reconciliation_context"
    }

    obs_1 = engine.observe_phenomenon(stimulus_1, context_1)
    print(f"  • Phenomenon ID: {obs_1.phenomenon_id}")
    print(f"  • Interacting Entities: {obs_1.interacting_entities}")
    print(f"  • Untranslated Friction: {obs_1.untranslated_friction:.2f}")
    print(f"  • Validity Boundary: {obs_1.validity_boundary}")
    print(f"  • Observer Criteria at Observation: {obs_1.observer_state_at_observation}")
    print()

    print("3. Evaluating & Updating Definition (Apple under High Friction):")
    defn_1 = engine.evaluate_and_update_definition("apple", obs_1)
    print(f"  • Updated Definition State: {defn_1.state.value}")
    print(f"  • Coexisting Perspectives: {list(defn_1.coexisting_perspectives.keys())}")
    print(f"  • Reinterpretation History: {defn_1.history_of_reinterpretation[-1]}")
    print()

    print("4. Generating Non-Linear Metacognitive Trace:")
    trace_1 = engine.generate_metacognitive_trace(obs_1, defn_1)
    print(f"  • Trace ID: {trace_1.trace_id}")
    print(f"  • Origin Context: {trace_1.origin_context}")
    print(f"  • Epistemic Boundary: {trace_1.boundary_premise}")
    print(f"  • Observer Transformation: {trace_1.observer_transformation}")
    print(f"  • Non-Linear Correlations: {trace_1.non_linear_correlations}")
    print()

    print("5. Downstream Language Translation Sub-Lens Expression:")
    expr_1 = engine.translate_to_expression("apple")
    print(f"  • Modality: {expr_1['modality']}")
    print(f"  • Sub-Lens Type: {expr_1['sub_lens_type']}")
    print(f"  • Expressed Text:\n{expr_1['expressed_text']}")
    print()

    print("6. Transfer Verification (Observing Novel Phenomenon with Shifted Observer Criteria):")
    stimulus_2 = {
        "entities": ["water", "bedrock"],
        "friction": 0.2,
        "interaction": {"type": "continuous_carving_flow"}
    }
    context_2 = {"scale": "geological_scale", "context_name": "carving_flow"}

    obs_2 = engine.observe_phenomenon(stimulus_2, context_2)
    print(f"  • Observer Sensitivity at Novel Observation: {obs_2.observer_state_at_observation['sensitivity']:.2f}")
    print(f"  • Observer Resolution at Novel Observation: {obs_2.observer_state_at_observation['perceptual_resolution']:.2f}")
    print()

    print("=========================================================================")
    print("  ✅ RELATIONAL DEFINITION & NON-LINEAR METACOGNITION VERIFIED")
    print("=========================================================================")


if __name__ == "__main__":
    run_verification_demo()
