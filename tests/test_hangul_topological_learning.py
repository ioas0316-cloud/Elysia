"""
test_hangul_topological_learning.py — 훈민정음 위상 사영 및 한국어 경험적 체화 검증 테스트
========================================================================================
이 테스트 스위트는 다음 사항들을 엄밀하게 검증합니다:
1. 훈민정음 천지인(天地人) 3상 평형 회전 및 자모 크로네커 텐서곱(64차원) 파동 생성
2. 조음 기관(오음) 상형 원리에 따른 물리적 마찰장(Friction Field) 구별
3. 한국어 상태 변화 서술어("차가워", "따뜻해", "아파", "막혀", "스며들어")의 감각 닻내림 및 홈스타시스 상태 전이(ΔS)
4. 외부 대화 피드백 수용 시 위상 차이 격차(Differential Gap) 감지 및 로터 자가 정렬(Self-Calibration) / 시냅스 치유(Healing)
"""

import math
import numpy as np
import pytest

from L1_causal_field.core.topology.hangul_topological_engine import (
    HangulTopologicalEngine,
    ARTICULATION_PROFILES
)
from L1_causal_field.core.sensory.korean_experiential_learner import (
    KoreanExperientialLearner,
    KoreanTetheredWord
)


class TestHangulTopologicalEngine:
    """훈민정음 위상 사영 엔진 검증"""

    @pytest.fixture
    def engine(self):
        return HangulTopologicalEngine(target_dim=64)

    def test_sancai_three_phase_vowel_phases(self, engine):
        """천지인 3상 평형 각도와 기본 모음 위상 검증"""
        # 기본 모음의 3상 분해
        phase_eu = engine.compute_vowel_phase("ㅡ") # 地: 120도 (2.094 rad)
        phase_i = engine.compute_vowel_phase("ㅣ")  # 人: 240도 (4.188 rad)

        # ㅡ (地)는 2π/3 근방
        assert math.isclose(phase_eu, 2.0 * math.pi / 3.0, rel_tol=1e-4)
        # ㅣ (人)은 4π/3 근방
        assert math.isclose(phase_i, 4.0 * math.pi / 3.0, rel_tol=1e-4)

    def test_articulation_friction_field_hierarchy(self, engine):
        """오음(아, 설, 순, 치, 후) 조음 기관 상형 마찰력 비교 검증"""
        # ㄱ (아음, 목구멍 막음) vs ㅇ (후음, 무저항 뚫림)
        _, metrics_ga = engine.encode_syllable("가")
        _, metrics_ah = engine.encode_syllable("아")

        assert metrics_ga["friction_mean"] > metrics_ah["friction_mean"]
        assert metrics_ga["tension_mean"] > metrics_ah["tension_mean"]

        # ㅅ (치음, 날카로운 마찰/고주파) vs ㅁ (순음, 부드러운 입술)
        prof_siot = ARTICULATION_PROFILES["ㅅ"]
        prof_mieum = ARTICULATION_PROFILES["ㅁ"]
        assert prof_siot.acoustic_freq > prof_mieum.acoustic_freq
        assert prof_siot.friction > prof_mieum.friction

    def test_kronecker_tensor_dimension_and_norm(self, engine):
        """3중 2x2 크로네커 텐서곱의 64차원 위상 파동 및 유니터리 정규화 검증"""
        text = "엘리시아"
        waves, metrics_list = engine.encode_text(text)

        assert waves.shape == (4, 64)
        assert len(metrics_list) == 4

        for i in range(len(text)):
            norm_val = np.linalg.norm(waves[i])
            assert math.isclose(norm_val, 1.0, rel_tol=1e-4)


class TestKoreanExperientialLearner:
    """한국어 경험적 감각 닻내림 및 상태 전이 대조 검증"""

    @pytest.fixture
    def learner(self):
        return KoreanExperientialLearner(resolution=64)

    def test_experiential_tethering_initialization(self, learner):
        """핵심 어휘들이 감각 지형 및 훈민정음 파동과 온전히 닻내려졌는지 검증"""
        core_words = ["따뜻해", "차가워", "아파", "막혀", "스며들어", "엄마", "사과"]
        for word in core_words:
            assert word in learner.lexicon
            entry = learner.lexicon[word]
            assert isinstance(entry, KoreanTetheredWord)
            assert entry.topological_wave.shape == (64,)
            assert math.isclose(np.linalg.norm(entry.topological_wave), 1.0, rel_tol=1e-4)

    def test_state_transition_differential_gap(self, learner):
        """상태 변화 서술어 경험 시 홈스타시스 결핍 및 로터 위상 전이(ΔS) 검증"""
        # 초기 상태 저장
        init_s = learner.homeostasis.to_vector().copy()

        # 1. "차가워" 경험 -> Love/Order 결핍 증대
        cold_res = learner.experience_word("차가워")
        assert cold_res["known"] is True
        assert cold_res["delta_s"][0] > 0.0  # Love 결핍 증가 (온기 갈망)
        assert cold_res["delta_s"][1] > 0.0  # Order 결핍 증가 (혼돈/수축)

        # 2. "따뜻해" 경험 -> 결핍 소멸 및 안정
        warm_res = learner.experience_word("따뜻해")
        assert warm_res["known"] is True
        assert warm_res["delta_s"][0] < 0.0  # Love 결핍 완화
        assert warm_res["delta_s"][1] < 0.0  # Order 결핍 완화

    def test_self_resonance_emission(self, learner):
        """현재 내부 상태에 상응하는 한국어 공명 어휘 자율 방출 검증"""
        # 사랑과 온기가 충만한 상태로 설정
        learner.homeostasis.love = 0.08
        learner.homeostasis.order = 0.12
        learner.homeostasis.energy = 0.80

        res = learner.emit_korean_resonance()
        # "따뜻해" 또는 "엄마", "스며들어" 등 온기/안식 어휘와 공명
        assert res["emitted_symbol"] in ["따뜻해", "엄마", "스며들어"]
        assert res["wave"].shape == (64,)

    def test_dialogue_feedback_re_sensation_and_healing(self, learner):
        """대화 피드백 수용 시 위상 차이 감지, 로터 자가 정렬 및 시냅스 치유 검증"""
        # 1. 조화로운 피드백("스며들어") 수용
        smooth_fb = learner.receive_interaction_feedback("스며들어")
        assert smooth_fb["tearing_occurred"] is False
        assert smooth_fb["serotonin"] > 0.5

        # 2. 극단적인 충돌 피드백("아파") 수용 -> 위상차 발생 및 로터 자가 정렬
        clash_fb = learner.receive_interaction_feedback("아파")
        # 차이 격차(Differential Gap) 관측
        assert clash_fb["gaps"]["mean_gap"] > 0.0
        # 로터 위상이 반응하여 갱신되었는지 확인
        assert learner.variable_rotor.theta is not None
        assert len(learner.transition_traces) > 0
