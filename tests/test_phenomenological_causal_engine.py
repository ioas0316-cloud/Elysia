import pytest
import numpy as np
from core.topology.phenomenological_causal_engine import (
    DomainPrimitive,
    IntentVector,
    ContinuousPhenomenologicalField,
    PurposePreservingProjectionEngine,
    GinzburgLandauPhaseTransitionEngine,
    PhenomenologicalCausalEngine,
)


def test_phenomenological_field_consistency():
    field = ContinuousPhenomenologicalField(feature_dim=16)

    # Aligned state vectors
    v = np.ones(16) / 4.0
    p1 = DomainPrimitive(
        domain_name="code", component="trap", principle="override", arrangement="syntax", state_vector=v
    )
    p2 = DomainPrimitive(
        domain_name="physics", component="flux", principle="conservation", arrangement="field", state_vector=v
    )

    intent = IntentVector(
        target_goal="safety", process_path="drain", outcome_teleology="salvation", target_vector=v
    )

    c_total, c_iso, c_tele, f_cross = field.evaluate_total_consistency([p1, p2], intent)

    assert c_iso > 0.99
    assert c_tele > 0.99
    assert f_cross < 0.01
    assert c_total > 0.95


def test_purpose_preserving_projection_p3_feedback_loop():
    field = ContinuousPhenomenologicalField(feature_dim=16)
    p3_engine = PurposePreservingProjectionEngine(field, threshold=0.75)

    v_target = np.array([1.0] + [0.0] * 15)
    v_opposing = np.array([-1.0] + [0.0] * 15)

    p_code = DomainPrimitive(
        domain_name="code", component="lock", principle="safety_trap", arrangement="kernel",
        state_vector=v_opposing, is_hard_constraint=True
    )
    p_cognition = DomainPrimitive(
        domain_name="cognition", component="duty", principle="sacrifice", arrangement="moral",
        state_vector=v_target, is_hard_constraint=False
    )

    intent = IntentVector(
        target_goal="salvation", process_path="override", outcome_teleology="prevent_disaster",
        target_vector=v_target
    )

    res = p3_engine.resolve_friction_loop([p_code, p_cognition], intent)

    assert res["final_c_total"] > res["history"][0]["c_total"]
    assert res["final_f_cross"] < res["history"][0]["f_cross"]
    # Check that code constraint was relaxed into soft cost
    assert not res["resolved_primitives"][0].is_hard_constraint


def test_ginzburg_landau_phase_transition():
    engine = GinzburgLandauPhaseTransitionEngine(critical_temp=1.0, alpha=1.0, beta=0.5)

    v_intent = np.array([1.0] + [0.0] * 15)
    v_shock = np.array([1.0] + [0.0] * 15) * 2.0
    intent = IntentVector("goal", "path", "outcome", target_vector=v_intent)

    res = engine.simulate_phase_transition(
        initial_eta=0.1, temp=0.5, perturbation_vector=v_shock, intent=intent, steps=100
    )

    assert res["phase_transition_occurred"] is True
    assert res["final_order_parameter"] > res["initial_order_parameter"]


def test_phenomenological_causal_engine_integration():
    engine = PhenomenologicalCausalEngine(feature_dim=16)

    v_intent = np.ones(16) / 4.0
    v_code = np.array([-1.0] + [0.0] * 15)
    v_phys = np.ones(16) / 4.0

    p_code = DomainPrimitive("code", "trap", "lock", "kernel", state_vector=v_code, is_hard_constraint=True)
    p_phys = DomainPrimitive("physics", "pressure", "flux", "field", state_vector=v_phys, is_hard_constraint=False)

    intent = IntentVector("salvation", "valve_open", "prevent_explosion", target_vector=v_intent)
    v_shock = np.ones(16) * 1.5

    result = engine.process_causal_scenario([p_code, p_phys], intent, external_perturbation=v_shock)

    assert "initial_metrics" in result
    assert "p3_resolution" in result
    assert "phase_transition" in result
    assert result["phase_transition"]["phase_transition_occurred"] is True
