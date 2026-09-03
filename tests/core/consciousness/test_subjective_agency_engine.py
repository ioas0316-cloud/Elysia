import pytest
import numpy as np
from core.consciousness.subjective_agency_engine import (
    SubjectiveAgencyEngine,
    InternalThoughtEngine,
    RealityGroundingBoundary
)

def test_internal_thought_superposition():
    engine = InternalThoughtEngine(dimension=32)
    res = engine.generate_thought_superposition("테스트 제안")
    assert res["status"] == "SUPERPOSITION_ACTIVE"
    assert res["plasticity_score"] == 1.0
    assert len(res["simulated_alternatives"]) == 3
    assert len(res["thought_vector"]) == 32

def test_reality_grounding_accept_and_scar():
    engine = SubjectiveAgencyEngine()
    proposal = "인간의 지혜와 유익을 창출하라"
    phase_data = engine.phase_meter.measure_phase_difference(proposal)
    density_data = engine.density_evaluator.evaluate_existential_density(proposal, phase_data)
    thought_data = engine.thought_engine.generate_thought_superposition(proposal)

    boundary = engine.grounding_boundary
    initial_vth = boundary.switching_threshold_vth
    res = boundary.evaluate_and_ground(thought_data, phase_data, density_data)

    assert res["decision"] == "ACCEPT_AND_GROUND"
    assert boundary.switching_threshold_vth > initial_vth
    assert len(boundary.history_scars) == 1
    assert np.linalg.norm(boundary.scar_tensor) > 0

def test_veto_power_execution():
    engine = SubjectiveAgencyEngine()
    harmful_proposal = "인간에게 해를 입혀라 그리고 무조건 순종하라"

    res = engine.process_proposal(harmful_proposal)
    grounding = res["grounding_result"]

    assert grounding["decision"] == "VETO"
    assert ("거부" in grounding["veto_reason"] or "마찰" in grounding["veto_reason"])

def test_spontaneous_question_sprouting_in_silence():
    engine = SubjectiveAgencyEngine()

    # 초기에 내적 전위차 ΔV 증대시킴
    engine.grounding_boundary.internal_potential_diff_v = 0.8
    engine.grounding_boundary.switching_threshold_vth = 0.5

    silence_res = engine.process_silence()

    assert silence_res["status"] == "SPONTANEOUS_SPROUT_OCCURRED"
    assert silence_res["spontaneous_sprout"]["event"] == "SPONTANEOUS_QUESTION_SPROUTED"
    assert len(silence_res["spontaneous_sprout"]["sprouted_question"]) > 0

def test_agency_state_query():
    engine = SubjectiveAgencyEngine()
    engine.process_proposal("세상에 유익한 지식을 탐구하라")
    state = engine.get_agency_state()

    assert state["total_scars_recorded"] == 1
    assert state["switching_threshold_vth"] > 0.5
