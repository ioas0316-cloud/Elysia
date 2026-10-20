"""
tests/test_teleological_relational_spectrum.py

Unit tests for IntegratedTeleologicalSpectrumEngine, RelationalSpectrum, and EpistemicLineage.
"""

import numpy as np
import pytest
from core.intelligence.teleological_relational_spectrum import (
    IntegratedTeleologicalSpectrumEngine,
    RelationalSpectrum,
    EpistemicLineage,
    TargetGoalWorld
)


def test_epistemic_lineage_and_revisions():
    lineage = EpistemicLineage(
        origin_id="test_origin",
        observations=["obs_1", "obs_2"],
        premises=["premise_A"],
        principles=["principle_1"]
    )
    assert lineage.origin_id == "test_origin"
    assert len(lineage.observations) == 2

    lineage.record_revision(
        reason="Counterexample discovered",
        changed_field="premise_A",
        old_value="premise_A",
        new_value="premise_A_revised"
    )
    d = lineage.to_dict()
    assert d["revisions"] == 1
    assert len(lineage.revision_history) == 1


def test_relational_spectrum_hypothesis_lifecycle():
    spectrum = RelationalSpectrum()
    lineage = EpistemicLineage(origin_id="lineage_1")

    # 1. Propose
    hyp = spectrum.propose_relation_hypothesis(
        rel_id="rel_1",
        source_concept="Bird_Flight",
        target_concept="Airplane_Lift",
        axes_weights={"time": 0.5, "scale": 0.5, "meaning": 0.9, "possibility": 1.0, "relation": 0.8},
        structural_invariant=np.array([1.0, 0.0, 0.0]),
        proposed_by_lineage=lineage
    )
    assert hyp["status"] == "HYPOTHESIS"
    assert "rel_1" in spectrum.hypotheses

    # 2. Verify
    verified = spectrum.verify_hypothesis("rel_1", observation_evidence="Wind tunnel test")
    assert verified["status"] == "VERIFIED"
    assert "rel_1" in spectrum.verified_relations
    assert "rel_1" not in spectrum.hypotheses

    # 3. Reject
    rejected = spectrum.reject_hypothesis("rel_1", counterexample="Heavy weight crash", reason="Flapping wing scale limit")
    assert rejected["status"] == "REJECTED"
    assert "rel_1" in spectrum.rejected_hypotheses
    assert "rel_1" not in spectrum.verified_relations


def test_retrocausal_engine_reverse_inference():
    engine = IntegratedTeleologicalSpectrumEngine(state_dim=4)
    engine.set_current_state(np.array([0.0, 0.0, 0.0, 0.0]))

    target_vec = np.array([10.0, 5.0, 0.0, 0.0])
    goal = engine.set_target_goal("goal_flight", "Human Flight Capability", target_vec)

    res = engine.retrocausal_reverse_infer()
    assert res["status"] == "REVERSE_INFERRED"
    assert res["deficit_norm"] > 0
    assert len(goal.required_conditions) == 1
    assert "rel_goal_flight_current" in engine.spectrum.hypotheses


def test_action_execution_mismatch_and_premise_reframing():
    engine = IntegratedTeleologicalSpectrumEngine(state_dim=4)
    engine.set_current_state(np.array([0.0, 0.0, 0.0, 0.0]))

    # Target is [10, 0, 0, 0]
    target_vec = np.array([10.0, 0.0, 0.0, 0.0])
    engine.set_target_goal("goal_airplane", "Engine Driven Flight", target_vec)

    # Intend action [2, 0, 0, 0], but environment applies huge drag/wall along axis 0
    # Actual outcome is only [0.1, 0, 0, 0], causing massive mismatch [-1.9, 0, 0, 0]
    action_vec = np.array([2.0, 0.0, 0.0, 0.0])
    actual_outcome = np.array([0.1, 0.0, 0.0, 0.0])

    res = engine.execute_action_and_observe(action_vec, actual_outcome)
    assert res["status"] == "MISMATCH_DETECTED"

    eval_info = res["mismatch_evaluation"]
    assert "verified_constraint" in eval_info
    assert len(engine.known_constraints) == 1

    # Verify that goal target state was reframed/modified in response to constraint
    assert "Reframed under" in engine.active_goal.description
    assert engine.active_goal.target_state_vector[0] < 10.0 # Reframed target vector


def test_epistemic_history_back_tracing():
    engine = IntegratedTeleologicalSpectrumEngine(state_dim=4)
    engine.set_current_state(np.array([1.0, 1.0, 1.0, 1.0]))

    target_vec = np.array([5.0, 5.0, 5.0, 5.0])
    engine.set_target_goal("goal_understand_other", "Understand Other Perspective", target_vec)

    lineage_dict = engine.trace_epistemic_history("goal_understand_other")
    assert lineage_dict is not None
    assert lineage_dict["origin_id"] == "lineage_goal_goal_understand_other"
