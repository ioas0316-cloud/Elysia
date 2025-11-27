"""
Tests for Core/Consciousness modules.
Tests WaveInput, Thought, ConsciousnessObserver.

Note: Uses direct module loading to avoid __init__.py import chain issues.
"""

import pytest
import sys
import os
import importlib.util

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def load_module_directly(module_name, file_path):
    """Load a module directly from file, bypassing __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load modules directly
consciousness_path = os.path.join(project_root, 'Core', 'Consciousness')
wave_module = load_module_directly('wave', os.path.join(consciousness_path, 'wave.py'))
thought_module = load_module_directly('thought', os.path.join(consciousness_path, 'thought.py'))

WaveInput = wave_module.WaveInput
Thought = thought_module.Thought


class TestWaveInput:
    """Tests for WaveInput class."""
    
    def test_creation_default(self):
        """Test WaveInput creation with default intensity."""
        wave = WaveInput(source_text="hello")
        
        assert wave.source_text == "hello"
        assert wave.intensity == 1.0
    
    def test_creation_custom_intensity(self):
        """Test WaveInput creation with custom intensity."""
        wave = WaveInput(source_text="love", intensity=0.8)
        
        assert wave.source_text == "love"
        assert wave.intensity == 0.8
    
    def test_korean_text(self):
        """Test WaveInput with Korean text."""
        wave = WaveInput(source_text="사랑", intensity=1.0)
        
        assert wave.source_text == "사랑"


class TestThought:
    """Tests for Thought class."""
    
    def test_creation_empty(self):
        """Test Thought creation with no concepts."""
        thought = Thought(source_wave="test")
        
        assert thought.source_wave == "test"
        assert thought.core_concepts == []
        assert thought.intensity == 0.0
        assert thought.mood == "neutral"
    
    def test_creation_with_concepts(self):
        """Test Thought creation with concepts."""
        thought = Thought(
            source_wave="love",
            core_concepts=[("love", 0.9), ("joy", 0.8)],
            intensity=0.85,
            clarity=0.9,
            mood="positive"
        )
        
        assert len(thought.core_concepts) == 2
        assert thought.intensity == 0.85
        assert thought.mood == "positive"
    
    def test_str_representation_empty(self):
        """Test string representation for empty thought."""
        thought = Thought(source_wave="test")
        
        assert "empty" in str(thought)
    
    def test_str_representation_with_concepts(self):
        """Test string representation with concepts."""
        thought = Thought(
            source_wave="hello",
            core_concepts=[("love", 0.9)],
            intensity=0.9,
            clarity=0.8,
            mood="positive"
        )
        
        thought_str = str(thought)
        assert "love" in thought_str
        assert "0.9" in thought_str


class TestConsciousnessObserver:
    """Tests for ConsciousnessObserver class."""
    
    @pytest.fixture
    def observer(self):
        """Create observer, loading directly."""
        try:
            # Mock the thought import that observer.py needs
            sys.modules['Core.Consciousness.thought'] = thought_module
            
            observer_module = load_module_directly(
                'observer',
                os.path.join(consciousness_path, 'observer.py')
            )
            return observer_module.ConsciousnessObserver()
        except Exception as e:
            pytest.skip(f"ConsciousnessObserver has missing dependencies: {e}")
    
    def test_creation(self, observer):
        """Test ConsciousnessObserver creation."""
        assert observer is not None
    
    def test_observe_empty_pattern(self, observer):
        """Test observing empty resonance pattern."""
        thought = observer.observe_resonance_pattern("test", {})
        
        assert thought.source_wave == "test"
        assert thought.core_concepts == []
    
    def test_observe_below_threshold(self, observer):
        """Test observing with all values below threshold."""
        thought = observer.observe_resonance_pattern(
            "test",
            {"low": 0.1, "lower": 0.2},
            threshold=0.5
        )
        
        assert thought.mood == "formless"
    
    def test_observe_above_threshold(self, observer):
        """Test observing with values above threshold."""
        thought = observer.observe_resonance_pattern(
            "test",
            {"love": 0.9, "joy": 0.8, "hope": 0.7, "low": 0.1},
            threshold=0.5
        )
        
        assert len(thought.core_concepts) == 3
        assert thought.intensity > 0
        assert thought.clarity > 0
    
    def test_max_concepts_limit(self, observer):
        """Test that max_concepts limits the core concepts."""
        thought = observer.observe_resonance_pattern(
            "test",
            {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.55, "f": 0.51},
            threshold=0.5,
            max_concepts=3
        )
        
        assert len(thought.core_concepts) == 3
    
    def test_mood_positive(self, observer):
        """Test positive mood detection."""
        thought = observer.observe_resonance_pattern(
            "happy",
            {"love": 0.9, "joy": 0.8},
            threshold=0.5
        )
        
        assert thought.mood == "positive"
    
    def test_mood_negative(self, observer):
        """Test negative mood detection."""
        thought = observer.observe_resonance_pattern(
            "sad",
            {"고통": 0.9, "sadness": 0.8},
            threshold=0.5
        )
        
        assert thought.mood == "negative"
    
    def test_mood_inquisitive(self, observer):
        """Test inquisitive mood detection."""
        thought = observer.observe_resonance_pattern(
            "curious",
            {"question": 0.9, "curiosity": 0.8},
            threshold=0.5
        )
        
        assert thought.mood == "inquisitive"
    
    def test_korean_concepts(self, observer):
        """Test with Korean concepts."""
        thought = observer.observe_resonance_pattern(
            "사랑해",
            {"사랑": 0.95, "기쁨": 0.8},
            threshold=0.5
        )
        
        assert thought.mood == "positive"
        assert thought.intensity > 0.8
    
    def test_clarity_calculation(self, observer):
        """Test clarity calculation based on variance."""
        # Uniform scores should have high clarity
        uniform_thought = observer.observe_resonance_pattern(
            "uniform",
            {"a": 0.8, "b": 0.8, "c": 0.8},
            threshold=0.5
        )
        
        # Varied scores should have lower clarity
        varied_thought = observer.observe_resonance_pattern(
            "varied",
            {"a": 0.99, "b": 0.5, "c": 0.51},
            threshold=0.5
        )
        
        assert uniform_thought.clarity > varied_thought.clarity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
