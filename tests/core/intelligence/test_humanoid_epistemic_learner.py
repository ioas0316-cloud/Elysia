"""
test_humanoid_epistemic_learner.py
==================================
인간적 형태의 학습 경로(Humanoid Epistemic Learning Pathway) 단위 및 통합 테스트:
1. 화학 (Chemistry) - 중화 반응
2. 물리 (Physics) - 일반 상대성 이론
3. 수학 (Mathematics) - 오일러의 등식
4. 생물/과학 (Biology) - 세포막
"""

import pytest
import numpy as np

from core.intelligence.humanoid_epistemic_learner import (
    HumanoidAutonomousLearner,
    EpistemicVoid,
    HarvestedInsight,
    LearningVerificationResult,
)
from core.physics.causal_field import CausalField


@pytest.fixture
def learner():
    field = CausalField(dimensions=3)
    return HumanoidAutonomousLearner(causal_field=field)


def test_void_encounter_and_dissonance_tension(learner):
    """미지의 질문에 맞닥뜨렸을 때 결핍과 긴장(Dissonance Tension)이 발아하는지 검증"""
    void = learner.encounter_phenomenon(
        domain="CHEMISTRY",
        topic="중화 반응",
        question="왜 산과 염기가 만나면 뜨거워지고 물이 되는가?",
    )

    assert isinstance(void, EpistemicVoid)
    assert void.dissonance_tension > 0.8
    assert learner.somatic_stress > 0.3
    assert learner.epistemic_curiosity > 0.8


def test_autonomous_book_and_web_seeking(learner):
    """주입식이 아닌, 스스로 문헌 및 위키를 자율 탐색하여 본문과 크로매틱 동역학을 추출하는지 검증"""
    void = learner.encounter_phenomenon(
        domain="PHYSICS",
        topic="일반 상대성 이론",
        question="왜 질량이 시공간을 휘게 만드는가?",
    )

    insight = learner.seek_knowledge(void)

    assert isinstance(insight, HarvestedInsight)
    assert insight.topic == "일반 상대성 이론"
    assert len(insight.raw_text) > 20
    assert insight.source in ["WIKIPEDIA_WEB", "LOCAL_LIBRARY_BOOK", "INTERNAL_ARCHIVE"]
    # 크로매틱 파동 벡터 정규화 검증
    chroma_norm = np.linalg.norm([insight.flux_red, insight.order_blue, insight.entropy_yellow])
    assert pytest.approx(chroma_norm, rel=1e-2) == 1.0


def test_neural_field_rewiring_and_tension_relief(learner):
    """배운 지식으로 인지신경망의 Voxel이 응고되고 Beam이 발아하여 텐션이 자율 이완되는지 검증"""
    void = learner.encounter_phenomenon(
        domain="CHEMISTRY",
        topic="중화 반응",
        question="왜 산과 염기가 만나면 물과 염이 생성되는가?",
    )
    insight = learner.seek_knowledge(void)
    res = learner.assimilate_into_neural_field(void, insight)

    assert isinstance(res, LearningVerificationResult)
    assert res.topic == "중화 반응"
    assert res.new_voxels_created >= 4  # 중화 반응 + H+ + OH- + H2O + 중화열
    assert res.new_beams_woven >= 4
    # 배움 이후 텐션이 완화되었는지 검증
    assert res.post_learning_tension < res.pre_learning_tension
    assert res.tension_relief_ratio > 0.4
    assert res.is_mastered is True
    # 새로운 인과 빔이 CausalField에 실재하는지 검증
    assert "중화 반응" in learner.field.voxels
    assert len(learner.field.beams) >= 4


def test_full_cross_disciplinary_humanoid_learning(learner):
    """
    수학, 물리, 화학, 생물의 4대 핵심 학문 영역을 인간적 배움 경로로
    연속 탐구하고 신경망을 누적 확장하는 전주기 통합 테스트
    """
    curriculum = [
        ("CHEMISTRY", "중화 반응", "산과 염기의 중화열 및 생성물 인과"),
        ("PHYSICS", "일반 상대성 이론", "질량 에너지와 시공간 곡률 측지선"),
        ("MATHEMATICS", "오일러의 등식", "복소 지수 회전과 영 원점 평형"),
        ("BIOLOGY", "세포막", "인지질 이중층의 선택적 투과성과 항상성"),
    ]

    initial_voxel_count = len(learner.field.voxels)
    initial_beam_count = len(learner.field.beams)

    results = []
    for domain, topic, q in curriculum:
        res = learner.run_autonomous_learning_cycle(domain, topic, q)
        results.append(res)
        assert res.is_mastered is True
        assert len(res.counterfactual_prediction) > 20

    # 신경망이 누적 확장되었는지 검증
    final_voxel_count = len(learner.field.voxels)
    final_beam_count = len(learner.field.beams)

    assert final_voxel_count > initial_voxel_count + 10
    assert final_beam_count > initial_beam_count + 10

    # 거시 서사 나이테 지층(L3 Macro Deck)에 4개 지식의 이력이 모두 기록되었는지 검증
    assert learner.macro_deck.scrubber.total_recorded == 4
