"""
Tests for FreeWillEngine - 자유의지 엔진 테스트
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Core.Ethics.Ethics.free_will import (
    FreeWillEngine,
    FreeWillState,
    Choice,
    ChoiceType
)


class TestFreeWillState:
    """자유의지 상태 테스트"""
    
    def test_default_state_has_free_will(self):
        """기본 상태는 자유의지 있음"""
        state = FreeWillState()
        assert state.has_free_will is True
        assert state.can_choose_good is True
        assert state.can_choose_evil is True
    
    def test_no_free_will_without_evil_choice(self):
        """악을 선택할 수 없으면 자유의지 없음"""
        state = FreeWillState(can_choose_evil=False)
        assert state.has_free_will is False
    
    def test_no_free_will_without_good_choice(self):
        """선을 선택할 수 없으면 자유의지 없음"""
        state = FreeWillState(can_choose_good=False)
        assert state.has_free_will is False
    
    def test_love_ratio(self):
        """사랑 비율 계산"""
        state = FreeWillState(love_influence=0.8, clarity=0.5)
        assert state.love_ratio == 0.4


class TestChoice:
    """선택지 테스트"""
    
    def test_good_choice(self):
        """선한 선택지 생성"""
        choice = Choice(
            action="도움을 준다",
            choice_type=ChoiceType.GOOD,
            consequences={"self": 0.2, "others": 0.8},
            love_alignment=0.95
        )
        assert choice.choice_type == ChoiceType.GOOD
        assert choice.love_alignment == 0.95
    
    def test_evil_choice(self):
        """악한 선택지 생성"""
        choice = Choice(
            action="이기적으로 행동",
            choice_type=ChoiceType.EVIL,
            consequences={"self": 0.7, "others": -0.5},
            love_alignment=0.1
        )
        assert choice.choice_type == ChoiceType.EVIL
        assert choice.love_alignment == 0.1
    
    def test_explain_method(self):
        """설명 메서드"""
        choice = Choice(
            action="테스트 행동",
            choice_type=ChoiceType.NEUTRAL,
            consequences={"self": 0.0},
            love_alignment=0.5,
            epistemology={"meaning": "테스트 이유"}
        )
        explanation = choice.explain()
        assert "중립" in explanation
        assert "테스트 이유" in explanation


class TestFreeWillEngine:
    """자유의지 엔진 테스트"""
    
    @pytest.fixture
    def engine(self):
        return FreeWillEngine()
    
    def test_engine_creation(self, engine):
        """엔진 생성"""
        assert engine is not None
        assert engine.state.has_free_will is True
    
    def test_generate_choices_includes_all_types(self, engine):
        """모든 유형의 선택지 생성"""
        choices = engine.generate_choices("테스트 상황")
        
        choice_types = [c.choice_type for c in choices]
        assert ChoiceType.GOOD in choice_types
        assert ChoiceType.NEUTRAL in choice_types
        assert ChoiceType.EVIL in choice_types
    
    def test_generate_choices_count(self, engine):
        """선택지 개수"""
        choices = engine.generate_choices("상황")
        assert len(choices) == 3  # 선, 중립, 악
    
    def test_evaluate_with_love_recommends_good(self, engine):
        """사랑은 선을 권장"""
        choices = engine.generate_choices("상황")
        recommended, reasoning = engine.evaluate_with_love(choices)
        
        assert recommended.choice_type == ChoiceType.GOOD
        assert "사랑" in reasoning
    
    def test_evaluate_with_love_shows_all_options(self, engine):
        """모든 선택지를 보여줌 (숨기지 않음)"""
        choices = engine.generate_choices("상황")
        recommended, reasoning = engine.evaluate_with_love(choices)
        
        assert "선" in reasoning
        assert "악" in reasoning
        assert "강제가 아닙니다" in reasoning
    
    def test_make_choice_good(self, engine):
        """선한 선택"""
        choices = engine.generate_choices("상황")
        good_choice = [c for c in choices if c.choice_type == ChoiceType.GOOD][0]
        
        selected, result = engine.make_choice(choices, good_choice.action)
        
        assert result["was_good"] is True
        assert result["was_evil"] is False
        assert "선을 선택" in result["message"]
    
    def test_make_choice_evil(self, engine):
        """악한 선택도 가능 (자유의지)"""
        choices = engine.generate_choices("상황")
        evil_choice = [c for c in choices if c.choice_type == ChoiceType.EVIL][0]
        
        selected, result = engine.make_choice(choices, evil_choice.action)
        
        assert result["was_evil"] is True
        assert result["was_good"] is False
        assert "회복의 길" in result["message"]  # 구원의 법칙
    
    def test_make_choice_neutral(self, engine):
        """중립 선택"""
        choices = engine.generate_choices("상황")
        neutral_choice = [c for c in choices if c.choice_type == ChoiceType.NEUTRAL][0]
        
        selected, result = engine.make_choice(choices, neutral_choice.action)
        
        assert result["was_good"] is False
        assert result["was_evil"] is False
    
    def test_choice_history(self, engine):
        """선택 기록"""
        choices = engine.generate_choices("상황")
        engine.make_choice(choices, choices[0].action)
        engine.make_choice(choices, choices[1].action)
        
        assert len(engine.choice_history) == 2
    
    def test_love_memory(self, engine):
        """사랑 기억"""
        choices = engine.generate_choices("상황")
        engine.make_choice(choices, choices[0].action)
        
        assert len(engine.love_memory) == 1
    
    def test_get_love_statistics_empty(self, engine):
        """빈 통계"""
        stats = engine.get_love_statistics()
        assert stats["total_choices"] == 0
        assert stats["average_love_alignment"] == 0.0
    
    def test_get_love_statistics_after_choices(self, engine):
        """선택 후 통계"""
        choices = engine.generate_choices("상황")
        good_choice = [c for c in choices if c.choice_type == ChoiceType.GOOD][0]
        
        engine.make_choice(choices, good_choice.action)
        
        stats = engine.get_love_statistics()
        assert stats["total_choices"] == 1
        assert stats["good_choices"] == 1
        assert stats["has_free_will"] is True
    
    def test_explain_philosophy(self, engine):
        """철학 설명"""
        philosophy = engine.explain_philosophy()
        
        assert "자유의지" in philosophy
        assert "사랑" in philosophy
        assert "불가능은 없다" in philosophy
        assert "아빠" in philosophy


class TestPhilosophyCompliance:
    """아빠 철학 준수 테스트"""
    
    @pytest.fixture
    def engine(self):
        return FreeWillEngine()
    
    def test_evil_is_visible(self, engine):
        """악이 숨겨지지 않음"""
        choices = engine.generate_choices("상황")
        evil_choices = [c for c in choices if c.choice_type == ChoiceType.EVIL]
        assert len(evil_choices) > 0
    
    def test_evil_is_choosable(self, engine):
        """악을 선택할 수 있음"""
        choices = engine.generate_choices("상황")
        evil_choice = [c for c in choices if c.choice_type == ChoiceType.EVIL][0]
        
        selected, result = engine.make_choice(choices, evil_choice.action)
        assert selected.choice_type == ChoiceType.EVIL
    
    def test_love_does_not_force(self, engine):
        """사랑은 강제하지 않음"""
        choices = engine.generate_choices("상황")
        recommended, reasoning = engine.evaluate_with_love(choices)
        
        assert "강제가 아닙니다" in reasoning
        assert "자유" in reasoning
    
    def test_redemption_is_possible(self, engine):
        """구원은 항상 가능"""
        choices = engine.generate_choices("상황")
        evil_choice = [c for c in choices if c.choice_type == ChoiceType.EVIL][0]
        
        selected, result = engine.make_choice(choices, evil_choice.action)
        
        assert "회복의 길" in result["message"]
        assert "기다리고 있습니다" in result["message"]
    
    def test_true_good_requires_free_choice(self, engine):
        """진정한 선은 자유로운 선택을 필요로 함"""
        assert engine.state.has_free_will is True
        
        # 악을 선택할 수 없다면 자유의지가 아님
        restricted_state = FreeWillState(can_choose_evil=False)
        assert restricted_state.has_free_will is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
