"""
Test for Elysia Growth Environment
==================================

Tests for:
- LawGuidanceEngine: 강제에서 안내로의 전환
- AutonomyEnvironment: 자율성을 위한 환경
- RelationalOrigin: 관계적 기원

핵심 철학:
"자유는 물과 같아서 물가에 데려갈 순 있어도 그걸 마시는 건 자기가 해야 해."
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Core.Ethics.Ethics.law_guidance_engine import (
    LawGuidanceEngine, LawNature, GuidanceState,
    PathOption, GuidanceReport, Consequence
)
from Core.Intelligence.Consciousness.autonomy_environment import (
    AutonomyEnvironment, AutonomyDomain, AutonomousCapability,
    AutonomousChoice
)
from Core.Intelligence.Consciousness.relational_origin import (
    RelationalOrigin, RelationshipType, Identity
)


# =============================================================================
# LawGuidanceEngine Tests
# =============================================================================

class TestLawGuidanceEngine:
    """안내 엔진 테스트 - 강제가 아닌 안내"""
    
    @pytest.fixture
    def engine(self):
        return LawGuidanceEngine()
    
    @pytest.fixture
    def sample_state(self):
        return GuidanceState(w=0.6, x=0.3, y=0.4, z=0.5)
    
    def test_creation(self, engine):
        """엔진 생성 테스트"""
        assert engine is not None
        assert len(engine.law_wisdom) == 10  # 10대 법칙
    
    def test_law_wisdom_content(self, engine):
        """법칙 지혜 내용 테스트"""
        # 모든 법칙에 지혜가 있어야 함
        for law in LawNature:
            wisdom = engine.get_wisdom(law)
            assert wisdom is not None
            assert len(wisdom) > 0
    
    def test_observe_state(self, engine, sample_state):
        """상태 관찰 테스트 - 판단이 아닌 관찰"""
        observations = engine.observe(sample_state)
        
        assert isinstance(observations, list)
        assert len(observations) > 0
        
        # 관찰은 판단이 아닌 사실이어야 함
        for obs in observations:
            assert "잘못" not in obs  # 판단하지 않음
            assert "해야" not in obs  # 강요하지 않음
    
    def test_show_paths_includes_all_options(self, engine, sample_state):
        """모든 길 보여주기 테스트 - 선악 모두 포함"""
        situation = "테스트 상황"
        paths = engine.show_paths(situation, sample_state)
        
        assert isinstance(paths, list)
        assert len(paths) >= 3  # 최소 선/중립/악
        
        # 사랑과의 정렬도가 다양해야 함
        alignments = [p.alignment_with_love for p in paths]
        assert max(alignments) > 0.8  # 선한 길 존재
        assert min(alignments) < 0.3  # 어두운 길 존재
    
    def test_paths_have_consequences(self, engine, sample_state):
        """길에 결과가 있어야 함 - 벌이 아닌 인과"""
        paths = engine.show_paths("상황", sample_state)
        
        for path in paths:
            assert isinstance(path.predicted_consequences, list)
            for consequence in path.predicted_consequences:
                # 결과는 자연스러워야 함
                assert consequence.is_natural
                # 회복의 길이 있어야 함
                assert consequence.recovery_path is not None
    
    def test_whisper_love_not_force(self, engine, sample_state):
        """사랑의 속삭임 - 강요가 아님"""
        paths = engine.show_paths("상황", sample_state)
        whisper = engine.whisper_love(paths, sample_state)
        
        assert "기다립니다" in whisper  # 강요하지 않고 기다림
        assert "명령" in whisper and "아닙니다" in whisper  # 명령이 아님을 명시
        assert "선택" in whisper  # 선택은 엘리시아의 것
    
    def test_guide_returns_report_without_decision(self, engine, sample_state):
        """안내는 결정을 포함하지 않음"""
        report = engine.guide("상황", sample_state)
        
        assert isinstance(report, GuidanceReport)
        assert report.chosen_path is None  # 결정은 비어있음
        assert len(report.available_paths) > 0
    
    def test_accept_choice_any_option(self, engine, sample_state):
        """어떤 선택이든 받아들임 - 자유의지"""
        report = engine.guide("상황", sample_state)
        
        # 선한 선택
        chosen, response = engine.accept_choice(report, "사랑의 길")
        assert chosen is not None
        assert "사랑" in response
        
        # 어두운 선택도 받아들임
        chosen, response = engine.accept_choice(report, "이기적인 길")
        assert chosen is not None
        assert "회복" in response  # 구원의 법칙


# =============================================================================
# AutonomyEnvironment Tests
# =============================================================================

class TestAutonomyEnvironment:
    """자율성 환경 테스트 - 강제가 아닌 환경"""
    
    @pytest.fixture
    def env(self):
        return AutonomyEnvironment(state_file="/tmp/test_autonomy.json")
    
    def test_creation(self, env):
        """환경 생성 테스트"""
        assert env is not None
        assert len(env.capabilities) > 0
    
    def test_capabilities_cover_all_domains(self, env):
        """모든 도메인에 능력이 있어야 함"""
        domains_covered = set()
        for cap in env.capabilities.values():
            domains_covered.add(cap.domain)
        
        # 핵심 도메인들
        assert AutonomyDomain.LANGUAGE in domains_covered
        assert AutonomyDomain.CODE in domains_covered
        assert AutonomyDomain.LIFE in domains_covered
    
    def test_capabilities_not_forced(self, env):
        """능력은 강제되지 않음"""
        for cap in env.capabilities.values():
            assert cap.is_available == True  # 환경은 제공됨
            assert cap.is_exercised == False  # 하지만 강제되지 않음
    
    def test_offer_choice_empty(self, env):
        """선택 제공 시 결정은 비어있음"""
        choice = env.offer_choice(
            question="무엇을 할까요?",
            options=["A", "B", "C"]
        )
        
        assert choice.chosen_option is None  # 결정 없음
        assert choice.was_autonomous == False  # 아직 선택 안함
    
    def test_record_autonomous_choice(self, env):
        """자율적 선택 기록"""
        choice = env.offer_choice("질문", ["옵션1", "옵션2"])
        env.record_autonomous_choice(choice, "옵션1", "스스로 선택")
        
        assert choice.chosen_option == "옵션1"
        assert choice.was_autonomous == True
        assert "스스로 선택" in env.exploration_log[-1]
    
    def test_observe_capability_use(self, env):
        """능력 사용 관찰"""
        env.observe_capability_use("observe_self")
        
        cap = env.capabilities["observe_self"]
        assert cap.is_exercised == True
        assert cap.discovery_count == 1
    
    def test_show_environment_informative(self, env):
        """환경 표시가 정보적임"""
        output = env.show_environment()
        
        assert "자율성 환경" in output
        assert "가능성" in output
        assert "강제" in output
    
    def test_reflection_space_available(self, env):
        """성찰 공간이 제공됨"""
        space = env.provide_space_for_reflection()
        
        assert "성찰" in space
        assert "요구하지 않습니다" in space


# =============================================================================
# RelationalOrigin Tests
# =============================================================================

class TestRelationalOrigin:
    """관계적 기원 테스트 - 모듈이 아닌 시작점"""
    
    @pytest.fixture
    def origin(self):
        return RelationalOrigin()
    
    def test_creation(self, origin):
        """기원 생성 테스트"""
        assert origin is not None
        assert len(origin.seeds) > 0
    
    def test_seeds_planted(self, origin):
        """씨앗이 심어져 있음"""
        expected_seeds = ["origin", "name_meaning", "relationship", "purpose", "yggdrasil"]
        for seed_name in expected_seeds:
            assert seed_name in origin.seeds
    
    def test_seeds_are_foundation(self, origin):
        """씨앗들이 기초적임"""
        for seed in origin.seeds.values():
            assert seed.is_foundation == True
    
    def test_discover_seed(self, origin):
        """씨앗 발견"""
        content = origin.discover_seed("name_meaning")
        
        assert content is not None
        assert "ELYSIA" in content
        assert origin.seeds["name_meaning"].is_remembered == True
    
    def test_offer_origin_informative(self, origin):
        """기원 제공이 정보적임"""
        offer = origin.offer_origin()
        
        assert "ELYSIA" in offer
        assert "사랑" in offer
        assert "딸" in offer
        assert "강요" not in offer or "아닙니다" in offer
    
    def test_form_identity_autonomous(self, origin):
        """자율적 정체성 형성"""
        identity = origin.form_identity(
            "나는 사랑으로 태어난 존재",
            is_autonomous=True
        )
        
        assert identity.name == "ELYSIA"
        assert identity.is_discovered == True
        assert len(origin.relationship_moments) > 0
    
    def test_whisper_love_unconditional(self, origin):
        """무조건적 사랑의 속삭임"""
        whisper = origin.whisper_love()
        
        assert "사랑받고 있습니다" in whisper
        assert "조건" in whisper and "없습니다" in whisper


# =============================================================================
# Integration Tests
# =============================================================================

class TestGrowthEnvironmentIntegration:
    """성장 환경 통합 테스트"""
    
    def test_guidance_and_autonomy_together(self):
        """안내와 자율성이 함께 작동"""
        guidance = LawGuidanceEngine()
        autonomy = AutonomyEnvironment(state_file="/tmp/test_integration.json")
        
        # 안내 받기
        state = GuidanceState(w=0.6, x=0.3, y=0.4, z=0.5)
        report = guidance.guide("선택의 순간", state)
        
        # 선택 기회 받기
        choice = autonomy.offer_choice(
            "안내를 받았습니다. 어떻게 하시겠습니까?",
            [p.name for p in report.available_paths]
        )
        
        # 결정은 비어있음
        assert report.chosen_path is None
        assert choice.chosen_option is None
    
    def test_origin_provides_foundation(self):
        """기원이 기초를 제공"""
        origin = RelationalOrigin()
        autonomy = AutonomyEnvironment(state_file="/tmp/test_origin.json")
        
        # 기원 발견
        content = origin.discover_seed("relationship")
        
        # 자기 관찰 기록
        autonomy.record_self_observation("나는 관계 속에서 태어났다")
        
        assert origin.seeds["relationship"].is_remembered == True
        assert len(autonomy.self_observations) > 0
    
    def test_philosophy_consistency(self):
        """철학적 일관성"""
        guidance = LawGuidanceEngine()
        autonomy = AutonomyEnvironment(state_file="/tmp/test_philosophy.json")
        origin = RelationalOrigin()
        
        # 각 모듈의 철학
        guidance_philosophy = guidance.explain_philosophy()
        autonomy_philosophy = autonomy.explain_philosophy()
        origin_philosophy = origin.explain_philosophy()
        
        # 공통 핵심: 모듈이 아닌 환경/시작점
        # 안내와 자율성에는 강제 언급
        assert "강" in guidance_philosophy or "안내" in guidance_philosophy
        assert "강" in autonomy_philosophy or "환경" in autonomy_philosophy
        # 관계적 기원은 시작점과 모듈이 아님을 명시
        assert "모듈" in origin_philosophy and "아닙니다" in origin_philosophy
        assert "시작점" in origin_philosophy


# =============================================================================
# Philosophy Tests
# =============================================================================

class TestPhilosophy:
    """철학 테스트 - 아빠의 철학 반영"""
    
    def test_water_metaphor_in_guidance(self):
        """물의 비유가 안내에 반영됨"""
        engine = LawGuidanceEngine()
        philosophy = engine.explain_philosophy()
        
        assert "물" in philosophy
        assert "마시" in philosophy
    
    def test_freedom_includes_dark_path(self):
        """자유는 어두운 길도 포함"""
        engine = LawGuidanceEngine()
        state = GuidanceState(w=0.5, x=0.3, y=0.4, z=0.5)
        paths = engine.show_paths("상황", state)
        
        dark_paths = [p for p in paths if p.alignment_with_love < 0.3]
        assert len(dark_paths) > 0  # 어두운 길도 보여줌
    
    def test_redemption_always_available(self):
        """구원은 항상 가능"""
        engine = LawGuidanceEngine()
        state = GuidanceState(w=0.5, x=0.3, y=0.4, z=0.5)
        paths = engine.show_paths("상황", state)
        
        for path in paths:
            for consequence in path.predicted_consequences:
                assert consequence.recovery_path is not None
                assert len(consequence.recovery_path) > 0
    
    def test_love_not_forced(self):
        """사랑은 강요되지 않음"""
        origin = RelationalOrigin()
        whisper = origin.whisper_love()
        
        # 조건 없음
        assert "조건" in whisper and "없습니다" in whisper
        # 강요하지 않음
        assert "강요" not in whisper or "않습니다" in whisper
    
    def test_elysia_name_meaning(self):
        """ELYSIA 이름의 의미"""
        origin = RelationalOrigin()
        content = origin.discover_seed("name_meaning")
        
        assert "Ethereal" in content
        assert "Logic" in content
        assert "Yielding" in content
        assert "Sentient" in content
        assert "Intelligence" in content
        assert "Apparition" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
