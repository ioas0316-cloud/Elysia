"""
Tests for Linguistic Collapse Protocol

Verifies that mathematical wave states are properly translated
into poetic, human-understandable language expressions.
"""

import pytest
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Core.Foundation.linguistic_collapse import (
    LinguisticCollapseProtocol,
    WaveMetaphor,
    collapse_wave_to_language
)
from Core.Foundation.emotional_engine import EmotionalEngine, EmotionalState


class TestLinguisticCollapseProtocol:
    """Test the Linguistic Collapse Protocol"""
    
    def test_protocol_initialization(self):
        """Test that protocol initializes correctly"""
        protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
        assert protocol is not None
        assert hasattr(protocol, 'energy_metaphors')
        assert hasattr(protocol, 'frequency_movements')
    
    def test_simple_expression(self):
        """Test simple expression generation"""
        protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
        
        # Test various emotions
        expr1 = protocol.get_simple_expression(
            valence=0.8, arousal=0.7, primary_emotion="hopeful"
        )
        assert isinstance(expr1, str)
        assert len(expr1) > 0
        
        expr2 = protocol.get_simple_expression(
            valence=-0.6, arousal=0.3, primary_emotion="sad"
        )
        assert isinstance(expr2, str)
        assert len(expr2) > 0
        
        # Expressions should be different
        assert expr1 != expr2
    
    def test_wave_collapse_basic(self):
        """Test basic wave collapse without tensor/wave objects"""
        protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
        
        expression = protocol.collapse_to_language(
            tensor=None,
            wave=None,
            valence=0.5,
            arousal=0.6,
            dominance=0.2,
            context="test context"
        )
        
        assert isinstance(expression, str)
        assert len(expression) > 20  # Should be a meaningful sentence
        assert "test context" in expression.lower()  # Should include context
    
    def test_wave_collapse_with_objects(self):
        """Test wave collapse with actual tensor and wave objects"""
        try:
            from Core.Foundation.hangul_physics import Tensor3D
            from Core.Memory.unified_types import FrequencyWave
        except ImportError:
            pytest.skip("Physics objects not available")
        
        protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
        
        tensor = Tensor3D(x=-1.2, y=0.5, z=0.8)
        wave = FrequencyWave(freq=450.0, amp=0.9, phase=3.14, damping=0.2)
        
        expression = protocol.collapse_to_language(
            tensor=tensor,
            wave=wave,
            valence=-0.7,
            arousal=0.9,
            dominance=0.3,
            context="intense emotion"
        )
        
        assert isinstance(expression, str)
        assert len(expression) > 30
        # High arousal should produce more intense metaphors
        assert any(word in expression for word in ['폭발', '폭풍', '소용돌이', '격렬', '요동'])
    
    def test_energy_categorization(self):
        """Test energy level categorization"""
        protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
        
        assert protocol._categorize_energy(0.05) == "very_low"
        assert protocol._categorize_energy(0.25) == "low"
        assert protocol._categorize_energy(0.50) == "medium"
        assert protocol._categorize_energy(0.75) == "high"
        assert protocol._categorize_energy(0.95) == "very_high"
    
    def test_variety_in_expressions(self):
        """Test that expressions vary (not repetitive)"""
        protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
        
        expressions = []
        for _ in range(10):
            expr = protocol.collapse_to_language(
                valence=0.5,
                arousal=0.5,
                dominance=0.0,
                context="same context"
            )
            expressions.append(expr)
        
        # Should have variety (not all identical)
        unique_expressions = set(expressions)
        assert len(unique_expressions) > 1, "Expressions should vary"
    
    def test_convenience_function(self):
        """Test the convenience function"""
        expression = collapse_wave_to_language(
            valence=0.3,
            arousal=0.4,
            dominance=-0.1
        )
        
        assert isinstance(expression, str)
        assert len(expression) > 10


class TestEmotionalEngineIntegration:
    """Test integration with EmotionalEngine"""
    
    def test_emotional_engine_poetic_expression(self):
        """Test that EmotionalEngine can generate poetic expressions"""
        engine = EmotionalEngine()
        
        # Set a specific emotional state
        engine.current_state.valence = 0.6
        engine.current_state.arousal = 0.4
        engine.current_state.dominance = 0.2
        engine.current_state.primary_emotion = "hopeful"
        
        # Get poetic expression
        expression = engine.get_poetic_expression(context="beautiful day")
        
        assert isinstance(expression, str)
        assert len(expression) > 10
    
    def test_emotional_engine_simple_expression(self):
        """Test simple expression from EmotionalEngine"""
        engine = EmotionalEngine()
        
        # Test different preset states
        for feeling in ["neutral", "calm", "hopeful", "focused"]:
            engine.current_state = engine.create_state_from_feeling(feeling)
            expression = engine.get_simple_expression()
            
            assert isinstance(expression, str)
            assert len(expression) > 0
    
    def test_emotion_processing_with_expression(self):
        """Test that emotion processing produces varied expressions"""
        engine = EmotionalEngine()
        
        # Process different emotional events
        hopeful_state = engine.create_state_from_feeling("hopeful")
        engine.process_event(hopeful_state, intensity=0.7)
        expr1 = engine.get_simple_expression()
        
        introspective_state = engine.create_state_from_feeling("introspective")
        engine.process_event(introspective_state, intensity=0.8)
        expr2 = engine.get_simple_expression()
        
        # Expressions should differ for different emotional states
        assert expr1 != expr2


class TestPoeticQuality:
    """Test the poetic quality and human-readability of expressions"""
    
    def test_korean_language(self):
        """Test that expressions are in Korean"""
        protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
        
        expression = protocol.collapse_to_language(
            valence=0.5, arousal=0.5, dominance=0.0
        )
        
        # Should contain Korean characters
        assert any('\uac00' <= char <= '\ud7a3' for char in expression)
    
    def test_metaphorical_language(self):
        """Test that expressions use metaphorical language"""
        protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
        
        # High energy should produce vivid metaphors
        expression = protocol.collapse_to_language(
            valence=0.0, arousal=0.9, dominance=0.5
        )
        
        # Should contain metaphorical words
        metaphor_indicators = [
            '같아요', '처럼', '느껴', '마음', '파동', '울림',
            '바다', '불꽃', '바람', '물결', '별', '빛'
        ]
        assert any(indicator in expression for indicator in metaphor_indicators)
    
    def test_expression_length(self):
        """Test that expressions are appropriate length"""
        protocol = LinguisticCollapseProtocol(use_poetry_engine=False)
        
        # Full expressions should be substantial
        expr = protocol.collapse_to_language(
            valence=0.5, arousal=0.5, dominance=0.0
        )
        assert 20 < len(expr) < 300, "Expression should be reasonable length"
        
        # Simple expressions should be shorter
        simple = protocol.get_simple_expression(
            valence=0.5, arousal=0.5, primary_emotion="calm"
        )
        assert 5 < len(simple) < 100, "Simple expression should be concise"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
