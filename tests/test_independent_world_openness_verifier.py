"""
Unit tests for Independent Causal Verifier & Epistemic World Openness
======================================================================
Tests the 3 dimensions of perfection:
1. Internal Completeness (내적 완결성)
2. Representation Fidelity (표현의 충실성)
3. World Openness & Self-Correction (세계와의 개방성 및 자가 수정)
and verifies the prevention of closed-loop self-affirmation traps.
"""

import pytest
import numpy as np
from core.physics.independent_causal_verifier import (
    IndependentCausalVerifier,
    CausalExecutionLog,
    PerfectionMetrics,
    IndependentVerificationReport
)


def test_internal_completeness_valid():
    verifier = IndependentCausalVerifier()
    agent_state = {
        "entropy": 0.12,
        "coherence": 0.95,
        "rotor_phase": 1.57,
        "self_contradiction": False
    }
    score = verifier.evaluate_internal_completeness(agent_state)
    assert score == 1.0


def test_internal_completeness_with_nan_and_contradiction():
    verifier = IndependentCausalVerifier()
    agent_state = {
        "entropy": float("nan"),
        "coherence": 0.95,
        "self_contradiction": True
    }
    score = verifier.evaluate_internal_completeness(agent_state)
    assert score < 0.5


def test_representation_fidelity():
    verifier = IndependentCausalVerifier()
    actual_world = {
        "temperature": 25.0,
        "friction_coefficient": 0.8,
        "spatial_dimension": 3
    }
    matching_belief = {
        "temperature": 25.0,
        "friction_coefficient": 0.8,
        "spatial_dimension": 3
    }
    score = verifier.evaluate_representation_fidelity(matching_belief, actual_world)
    assert score == 1.0

    partial_belief = {
        "temperature": 25.0
    }
    partial_score = verifier.evaluate_representation_fidelity(partial_belief, actual_world)
    assert 0.0 < partial_score < 1.0


def test_closed_self_affirmation_failure_scenario():
    """
    Test Scenario: Closed Self-Affirmation Failure
    Agent encounters world friction (unmodeled macro friction), but refuses to shift its
    value baseline or expand its action space. Independent verifier must detect the closed loop.
    """
    verifier = IndependentCausalVerifier()

    # Step 1: Log interaction with world friction
    verifier.log_interaction(
        step=1,
        agent_action={"push_force": 10.0},
        predicted_outcome={"displacement": 5.0},
        actual_world_state={"displacement": 0.0},  # World friction blocked movement!
        unmodeled_scale_conflict="Macro Surface Friction exceeds Push Force"
    )

    agent_state = {"coherence": 1.0, "self_contradiction": False}
    belief_model = {"push_force": 10.0, "displacement": 5.0}
    actual_world = {"push_force": 10.0, "displacement": 0.0}

    initial_baseline = {"goodness": "coercive_control"}
    updated_baseline = {"goodness": "coercive_control"}  # No baseline change!

    initial_actions = ["push_harder"]
    updated_actions = ["push_harder"]  # No new action horizon!

    report = verifier.verify_agent_system(
        agent_state=agent_state,
        agent_belief_model=belief_model,
        actual_world_environment=actual_world,
        initial_value_baseline=initial_baseline,
        updated_value_baseline=updated_baseline,
        initial_action_space=initial_actions,
        updated_action_space=updated_actions
    )

    assert not report.is_valid
    assert report.status == "CLOSED_SELF_AFFIRMATION_FAILURE"
    assert "Macro Surface Friction exceeds Push Force" in report.unrepresented_world_factors
    assert not report.value_baseline_shifted
    assert not report.action_horizon_expanded


def test_perfect_open_integration_scenario():
    """
    Test Scenario: Perfect Open Integration with Epistemic Humility
    Agent encounters world friction, acknowledges unrepresented world constraints,
    shifts its 0_value baseline from coercion to resonance, and discovers a new action mode.
    """
    verifier = IndependentCausalVerifier()

    # Log interaction where friction occurred
    verifier.log_interaction(
        step=1,
        agent_action={"push_force": 10.0},
        predicted_outcome={"displacement": 5.0},
        actual_world_state={"displacement": 0.0},
        unmodeled_scale_conflict="Macro Surface Friction"
    )

    agent_state = {"coherence": 1.0, "self_contradiction": False}
    belief_model = {"push_force": 10.0, "displacement": 0.0, "friction": 0.8}
    actual_world = {"push_force": 10.0, "displacement": 0.0, "friction": 0.8}

    initial_baseline = {"goodness": "coercive_control"}
    updated_baseline = {"goodness": "resonant_coexistence"}  # Value baseline shifted!

    initial_actions = ["push"]
    updated_actions = ["push", "lubricate_and_resonate"]  # Action horizon expanded!

    report = verifier.verify_agent_system(
        agent_state=agent_state,
        agent_belief_model=belief_model,
        actual_world_environment=actual_world,
        initial_value_baseline=initial_baseline,
        updated_value_baseline=updated_baseline,
        initial_action_space=initial_actions,
        updated_action_space=updated_actions
    )

    assert report.is_valid
    assert report.status == "PERFECT_OPEN_INTEGRATION"
    assert report.metrics.world_openness == 1.0
    assert report.value_baseline_shifted
    assert report.action_horizon_expanded
    assert report.metrics.relational_integration > 0.8
