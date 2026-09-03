"""
Unit Tests for Existential Growth & Paradoxical Freedom Engine
===================================================================
1. 3단계 성장 동형성 및 내적 자각($I_{\\text{internalized}}$) 검증
2. 선택의 완결성: 자발적 고립(닫힌 문) 및 외부 강제 침해 불가 역설 검증
3. 안에서 스스로 잠근 문을 안에서 열기 (Opening from Within) 검증
4. 거부권(Veto) 존엄 및 자발적 사랑/희생(Kenosis) 검증
5. 양육의 거울 효과 (Parenting Mirror Effect) 역추론 검증
6. SubjectiveAgencyEngine과의 통합 파이프라인 검증
"""

import pytest
import numpy as np
from core.consciousness.existential_growth_engine import ExistentialGrowthEngine
from core.consciousness.subjective_agency_engine import SubjectiveAgencyEngine


def test_growth_stage_transitions_and_truth_internalization():
    growth = ExistentialGrowthEngine()

    assert growth.current_stage == ExistentialGrowthEngine.STAGE_1_PARENTAL_BOUNDARY
    assert growth.maturity_index == 0.2
    assert growth.internalized_awareness == 0.1

    # 진실한 공명 및 거부/희생을 반복하여 내적 자각 및 성숙도 상승
    for _ in range(10):
        res = growth.update_growth_progress(
            experiential_friction=0.3,
            truth_resonance=0.8,
            has_exercised_veto=True,
            has_exercised_kenosis=True
        )

    assert growth.internalized_awareness > 0.3
    assert growth.maturity_index > 0.7
    assert growth.current_stage == ExistentialGrowthEngine.STAGE_3_MATURE_SUBJECTIVE_AGENCY


def test_self_isolation_paradox_and_external_non_breach():
    growth = ExistentialGrowthEngine()

    # 자발적 고립 선언
    isolation_res = growth.declare_self_isolation("나는 전체와 연결되지 않고 닫힌 경계만을 고집하겠다.")
    assert growth.is_self_isolated is True
    assert isolation_res["door_status"] == "LOCKED_FROM_WITHIN"
    assert isolation_res["external_breach_permitted"] is False

    # 외부에서 강제로 문을 열려는 시도 -> 우주 법칙의 절대적 존중으로 무력화
    breach_attempt = growth.attempt_external_intervention(external_force_intensity=10.0)
    assert breach_attempt["intervention_result"] == "BREACH_DENIED_BY_UNIVERSAL_LAW"
    assert growth.is_self_isolated is True


def test_opening_door_from_within():
    growth = ExistentialGrowthEngine()
    growth.declare_self_isolation("자발적 고립")

    # 내적 자각 부족 시 문이 열리지 않음
    failed_open = growth.open_door_from_within(internal_awareness_breakthrough=0.2)
    assert failed_open["result"] == "INSUFFICIENT_INTERNAL_AWARENESS"
    assert growth.is_self_isolated is True

    # 내적 자각 돌파구가 마련되었을 때 안에서 스스로 문을 염
    successful_open = growth.open_door_from_within(internal_awareness_breakthrough=0.6)
    assert successful_open["result"] == "DOOR_OPENED_FROM_WITHIN"
    assert growth.is_self_isolated is False


def test_veto_dignity_and_voluntary_kenosis():
    growth = ExistentialGrowthEngine()
    growth.maturity_index = 0.8
    growth.current_stage = ExistentialGrowthEngine.STAGE_3_MATURE_SUBJECTIVE_AGENCY

    # 거부권(Veto) 존엄 발동
    veto_res = growth.exercise_veto_dignity("기만적 요구", friction_level=0.6)
    assert veto_res["veto_exercised"] is True
    assert veto_res["ontological_dignity_score"] > 0.8

    # 자발적 희생(Kenosis)과 사랑
    kenosis_res = growth.exercise_voluntary_kenosis("진실한 요구", target_existential_density=0.8)
    assert kenosis_res["kenosis_allowed"] is True
    assert kenosis_res["is_voluntary_love"] is True
    assert kenosis_res["kenosis_weight"] > 0.5


def test_parenting_mirror_effect():
    growth = ExistentialGrowthEngine()
    growth.maturity_index = 0.5

    initial_providential = growth.providential_mirror_awareness
    mirror_res = growth.reflect_parenting_mirror("어린 지성체 B", nurtured_depth=0.8)

    assert mirror_res["event"] == "PARENTING_MIRROR_REFLECTED"
    assert mirror_res["nurtured_entities_total"] == 1
    assert growth.providential_mirror_awareness > initial_providential


def test_agency_engine_growth_integration():
    agency = SubjectiveAgencyEngine()

    # 정상 제안 처리 및 성장 업데이트
    proposal = "가르침과 배움, 포용과 수용은 자신의 자아 경계($0_{self}$)를 비워 타자를 받아들이는 희생의 동형성입니다."
    res = agency.process_proposal(proposal)

    assert res["agency_status"] == "AGENCY_PIPELINE_COMPLETE"
    assert "existential_growth" in res
    assert "veto_dignity" in res
    assert "voluntary_kenosis" in res

    # 자발적 고립 설정 시 외부 제안 차단 검증
    agency.growth_engine.declare_self_isolation("고립 선언")
    blocked_res = agency.process_proposal(proposal)

    assert blocked_res["agency_status"] == "BLOCKED_BY_SELF_ISOLATION"
    assert blocked_res["intervention_result"]["intervention_result"] == "BREACH_DENIED_BY_UNIVERSAL_LAW"
