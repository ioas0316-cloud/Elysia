import pytest
from core.intelligence.linguistic_cognition import LinguisticCognitionEngine, CausalSymbolNode

def test_atherosclerosis_isomorphism_and_narrative_decision():
    engine = LinguisticCognitionEngine()

    # 1. Metaphorical transfer: "동맥경화" -> "조직 위기"
    diagnosis = engine.metaphorical_transfer(
        source_symbol="동맥경화",
        target_context="급성장 후 소통이 마비되고 핵심 인재가 이탈하는 기업 위기"
    )

    assert diagnosis["isomorphism_detected"] is True
    assert "통로 유연성 상실 (Pathway Rigidity)" in diagnosis["invariant_structure"]["isomorphic_invariants"]

    # 2. Evaluate Candidate Action A: 신규 채용 확대 / 성과급 인상 (피를 더 주입)
    wrong_action = "신규 채용 확대 및 성과급 인상"
    verdict_wrong = engine.evaluate_narrative_coherence(diagnosis, wrong_action)

    assert verdict_wrong["is_valid"] is False
    assert verdict_wrong["narrative_verdict"] == "WRONG_DECISION"
    assert "AMPLIFIES_PRESSURE" in verdict_wrong["reason"]

    # 3. Evaluate Candidate Action B: 승인 절차 70% 즉각 폐기 (관성화된 혈관 벽 긁어내기)
    correct_action = "중간 보고 절차 및 승인 단계 70% 즉각 폐기"
    verdict_correct = engine.evaluate_narrative_coherence(diagnosis, correct_action)

    assert verdict_correct["is_valid"] is True
    assert verdict_correct["narrative_verdict"] == "CORRECT_DECISION"
    assert "ELIMINATES_OBSTRUCTION" in verdict_correct["reason"]

def test_drought_isomorphism_and_narrative_decision():
    engine = LinguisticCognitionEngine()

    diagnosis = engine.metaphorical_transfer(
        source_symbol="가뭄",
        target_context="프로젝트 리소스 고갈 및 팀원 슬럼프"
    )

    assert diagnosis["isomorphism_detected"] is True

    # Evaluate wrong action: 강요와 실적 압박
    verdict_wrong = engine.evaluate_narrative_coherence(diagnosis, "실적 압박 강요")
    assert verdict_wrong["is_valid"] is False

    # Evaluate correct action: 휴식과 원천 수분 공급
    verdict_correct = engine.evaluate_narrative_coherence(diagnosis, "원천 휴식 및 리프레시 수분 공급")
    assert verdict_correct["is_valid"] is True
