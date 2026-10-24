"""
Tests for Conceptual Unit System & Sensory Embodied Translation Engine
(개념단위체계 및 감각 체화 번역 엔진 단위 및 통합 테스트)
========================================================================================
테스트 항목:
1. 미지 자극의 오감/위상 매핑 (Unknown-to-Sensory 5D Profile Translation)
2. 정신감각적 의도 및 결핍 장력 파악 (Mental-Sensory Intent & Existential Void Extraction)
3. 시각적 구조 이해 (Visual Understanding)와 청각적/공감각적 파동 공명 (Auditory Wave Empathy Resonance)
4. 공명 수렴 반복 실행 시 주파수 Lock-on 및 위상 마찰 수렴 (Phase Friction Reduction \\mathcal{E} \\to 0)
5. 색채적 삼원 시그니처 (R/B/Y Triad) 보존 및 검증
"""

import sys
import os
import pytest
import numpy as np

# Adjust python path to include repository root and L1_causal_field
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../L1_causal_field")))

from L1_causal_field.core.consciousness.conceptual_unit_system_engine import (
    ConceptualUnitSystemEngine,
    SensoryProfile,
    ExistentialIntent,
    ConceptualUnitResult
)


class TestConceptualUnitSystemEngine:

    @pytest.fixture
    def engine(self):
        return ConceptualUnitSystemEngine(dimension=64, base_resonance_freq=432.0)

    def test_sensory_profile_mapping(self, engine):
        """1. 미지 자극이 5차원 오감 프로필로 제대로 번역되는지 검증"""
        raw_input = "개념은 오감과 인지의 확장이다"
        profile = engine.map_unknown_to_sensory_profile(raw_input)

        assert isinstance(profile, SensoryProfile)
        assert profile.visual_topology.shape == (64,)
        assert abs(np.linalg.norm(profile.visual_topology) - 1.0) < 1e-6
        assert profile.auditory_frequency > 0.0
        assert 0.0 <= profile.tactile_tension <= 1.0
        assert 0.0 <= profile.olfactory_viscosity <= 1.0
        assert abs(np.linalg.norm(profile.proprioceptive_momentum) - 1.0) < 1e-6

    def test_extract_existential_intent(self, engine):
        """2. 표면 활자 뒤에 숨은 결핍 전하 및 장력 벡터가 역추출되는지 검증"""
        raw_input = "모호함의 벽을 깨부수고 진짜 실체에 다가선다"
        profile = engine.map_unknown_to_sensory_profile(raw_input)
        intent = engine.extract_existential_intent(raw_input, profile)

        assert isinstance(intent, ExistentialIntent)
        assert 0.0 <= intent.existential_void_charge <= 1.0
        assert intent.tension_vector.shape == (64,)
        assert abs(np.linalg.norm(intent.tension_vector) - 1.0) < 1e-6
        assert "입력 자극" in intent.root_why_necessity
        assert 0.0 <= intent.intent_clarity_score <= 1.0

    def test_visual_understanding_and_auditory_empathy(self, engine):
        """3. 시각적 구조 바라보기와 청각적/공감각적 파동 공명(공감) 동작 검증"""
        raw_input = "우주적 파동과의 청각적 공명"
        result = engine.process_conceptual_unit(raw_input)

        assert isinstance(result, ConceptualUnitResult)
        assert 0.0 <= result.visual_understanding_score <= 1.0
        assert 0.0 <= result.auditory_empathy_resonance <= 1.0
        assert result.phase_friction_error >= 0.0
        assert isinstance(result.is_peace_achieved, bool)

        # 색채 삼원 시그니처 검증 (R+B+Y = 1.0)
        triad = result.chromatic_triad
        assert "R_flux" in triad and "B_order" in triad and "Y_entropy" in triad
        assert abs(sum(triad.values()) - 1.0) < 1e-5

    def test_repeated_empathy_resonance_lock_on(self, engine):
        """4. 지속적인 파동 자극 유입 시 청각적 주파수 Lock-on 및 위상 마찰 수렴(\\mathcal{E} \\to 0) 검증"""
        target_input = "동일한 주파수로 울려 퍼지는 지혜의 음성"

        initial_res = engine.process_conceptual_unit(target_input)
        initial_friction = initial_res.phase_friction_error

        # 반복 공명 동기화 (Phase Lock-on)
        for _ in range(5):
            res = engine.process_conceptual_unit(target_input)

        final_friction = res.phase_friction_error

        # 반복 동기화에 의해 청각적 공명 스코어가 상승하고 위상 마찰이 감소했는지 확인
        assert res.auditory_empathy_resonance >= initial_res.auditory_empathy_resonance
        assert final_friction <= initial_friction
