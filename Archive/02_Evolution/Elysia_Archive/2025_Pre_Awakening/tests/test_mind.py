"""
Tests for Core/Mind modules.
Tests ConceptSphere, WillVector, MirrorLayer, EmotionalPalette.

Note: Uses direct module loading to avoid __init__.py import chain issues.
"""

import pytest
import sys
import os
import importlib.util
import time

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
mind_path = os.path.join(project_root, 'Core', 'Mind')

# Load concept_sphere module
try:
    concept_sphere_module = load_module_directly('concept_sphere', os.path.join(mind_path, 'concept_sphere.py'))
    ConceptSphere = concept_sphere_module.ConceptSphere
    WillVector = concept_sphere_module.WillVector
    MirrorLayer = concept_sphere_module.MirrorLayer
    CONCEPT_SPHERE_AVAILABLE = True
except Exception as e:
    CONCEPT_SPHERE_AVAILABLE = False
    ConceptSphere = None
    WillVector = None
    MirrorLayer = None

# Load emotional_palette module
try:
    emotional_palette_module = load_module_directly('emotional_palette', os.path.join(mind_path, 'emotional_palette.py'))
    EmotionalPalette = emotional_palette_module.EmotionalPalette
    EmotionalSpectrum = emotional_palette_module.EmotionalSpectrum
    EMOTIONAL_PALETTE_AVAILABLE = True
except Exception as e:
    EMOTIONAL_PALETTE_AVAILABLE = False
    EmotionalPalette = None
    EmotionalSpectrum = None


class TestWillVector:
    """Tests for WillVector class."""
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_creation_default(self):
        """Test WillVector creation with defaults."""
        will = WillVector()
        
        assert will.x == 0.0
        assert will.y == 0.0
        assert will.z == 0.0
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_creation_with_values(self):
        """Test WillVector creation with values."""
        will = WillVector(x=0.5, y=0.3, z=0.8)
        
        assert will.x == 0.5
        assert will.y == 0.3
        assert will.z == 0.8
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_magnitude(self):
        """Test WillVector magnitude calculation."""
        will = WillVector(x=3.0, y=4.0, z=0.0)
        
        # 3-4-5 triangle
        assert abs(will.magnitude() - 5.0) < 0.001
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_magnitude_zero(self):
        """Test WillVector magnitude for zero vector."""
        will = WillVector()
        
        assert will.magnitude() == 0.0


class TestMirrorLayer:
    """Tests for MirrorLayer class."""
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_creation(self):
        """Test MirrorLayer creation."""
        mirror = MirrorLayer()
        
        assert mirror.phenomena == []
        assert mirror.intensity == 0.0
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_reflect(self):
        """Test reflecting world events."""
        mirror = MirrorLayer()
        
        mirror.reflect({"event": "sunrise", "location": "east"})
        
        assert len(mirror.phenomena) == 1
        assert mirror.phenomena[0]["event"] == "sunrise"
        assert "reflected_at" in mirror.phenomena[0]
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_reflect_multiple(self):
        """Test reflecting multiple events."""
        mirror = MirrorLayer()
        
        mirror.reflect({"event": "event1"})
        mirror.reflect({"event": "event2"})
        mirror.reflect({"event": "event3"})
        
        assert len(mirror.phenomena) == 3
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_project(self):
        """Test projecting phenomena."""
        mirror = MirrorLayer()
        
        mirror.reflect({"event": "test"})
        projected = mirror.project()
        
        assert len(projected) == 1
        assert projected[0]["event"] == "test"


class TestConceptSphere:
    """Tests for ConceptSphere class."""
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_creation(self):
        """Test ConceptSphere creation."""
        sphere = ConceptSphere("love")
        
        assert sphere.id == "love"
        assert sphere.parent is None
        assert sphere.activation_count == 0
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_activate(self):
        """Test ConceptSphere activation."""
        sphere = ConceptSphere("test")
        
        sphere.activate()
        
        assert sphere.activation_count == 1
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_activate_multiple(self):
        """Test multiple activations."""
        sphere = ConceptSphere("test")
        
        sphere.activate()
        sphere.activate()
        sphere.activate()
        
        assert sphere.activation_count == 3
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_add_sub_concept(self):
        """Test adding sub-concepts."""
        parent = ConceptSphere("love")
        child = parent.add_sub_concept("affection")
        
        assert "affection" in parent.sub_concepts
        assert child.parent == parent
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_set_emotion(self):
        """Test setting emotions."""
        sphere = ConceptSphere("test")
        
        sphere.set_emotion("joy", 0.8)
        sphere.set_emotion("peace", 0.6)
        
        assert sphere.emotions["joy"] == 0.8
        assert sphere.emotions["peace"] == 0.6
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_set_emotion_clamped(self):
        """Test emotion values are clamped to [0, 1]."""
        sphere = ConceptSphere("test")
        
        sphere.set_emotion("high", 1.5)
        sphere.set_emotion("low", -0.5)
        
        assert sphere.emotions["high"] == 1.0
        assert sphere.emotions["low"] == 0.0
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_set_value(self):
        """Test setting values."""
        sphere = ConceptSphere("test")
        
        sphere.set_value("truth", 0.9)
        sphere.set_value("beauty", 0.7)
        
        assert sphere.values["truth"] == 0.9
        assert sphere.values["beauty"] == 0.7
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_set_will(self):
        """Test setting will direction."""
        sphere = ConceptSphere("test")
        
        sphere.set_will(0.5, 0.3, 0.8)
        
        assert sphere.will.x == 0.5
        assert sphere.will.y == 0.3
        assert sphere.will.z == 0.8
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_get_slice(self):
        """Test getting concept slice."""
        sphere = ConceptSphere("test")
        sphere.set_will(1.0, 0.0, 0.0)
        sphere.set_emotion("joy", 0.8)
        sphere.activate()
        
        slice_data = sphere.get_slice()
        
        assert slice_data["id"] == "test"
        assert slice_data["will_magnitude"] == 1.0
        assert slice_data["activation_count"] == 1
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_to_dict(self):
        """Test serialization to dictionary."""
        sphere = ConceptSphere("love")
        sphere.set_emotion("joy", 0.9)
        sphere.add_sub_concept("affection")
        
        data = sphere.to_dict()
        
        assert data["id"] == "love"
        assert "joy" in data["emotions"]
        assert "affection" in data["sub_concepts"]
    
    @pytest.mark.skipif(not CONCEPT_SPHERE_AVAILABLE, reason="ConceptSphere module not available")
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "love",
            "will": {"x": 0.5, "y": 0.3, "z": 0.8},
            "emotions": {"joy": 0.9},
            "values": {"truth": 0.8},
            "language_tokens": ["love", "heart"],
            "activation_count": 5
        }
        
        sphere = ConceptSphere.from_dict(data)
        
        assert sphere.id == "love"
        assert sphere.will.x == 0.5
        assert sphere.emotions["joy"] == 0.9
        assert sphere.activation_count == 5


class TestEmotionalPalette:
    """Tests for EmotionalPalette class."""
    
    @pytest.mark.skipif(not EMOTIONAL_PALETTE_AVAILABLE, reason="EmotionalPalette module not available")
    def test_creation(self):
        """Test EmotionalPalette creation."""
        palette = EmotionalPalette()
        
        assert palette is not None
        assert "Joy" in palette.base_emotions
        assert "Sadness" in palette.base_emotions
    
    @pytest.mark.skipif(not EMOTIONAL_PALETTE_AVAILABLE, reason="EmotionalPalette module not available")
    def test_base_emotions_count(self):
        """Test that base emotions are loaded."""
        palette = EmotionalPalette()
        
        assert len(palette.base_emotions) >= 5
    
    @pytest.mark.skipif(not EMOTIONAL_PALETTE_AVAILABLE, reason="EmotionalPalette module not available")
    def test_mix_emotion_single(self):
        """Test mixing a single emotion."""
        palette = EmotionalPalette()
        
        qubit = palette.mix_emotion({"Joy": 1.0})
        
        assert qubit is not None
        # Qubit created successfully
        assert hasattr(qubit, 'state')
    
    @pytest.mark.skipif(not EMOTIONAL_PALETTE_AVAILABLE, reason="EmotionalPalette module not available")
    def test_mix_emotion_multiple(self):
        """Test mixing multiple emotions."""
        palette = EmotionalPalette()
        
        qubit = palette.mix_emotion({"Joy": 0.6, "Sadness": 0.4})
        
        assert qubit is not None
    
    @pytest.mark.skipif(not EMOTIONAL_PALETTE_AVAILABLE, reason="EmotionalPalette module not available")
    def test_mix_emotion_empty(self):
        """Test mixing with empty components."""
        palette = EmotionalPalette()
        
        qubit = palette.mix_emotion({})
        
        # Returns a neutral qubit
        assert qubit is not None
    
    @pytest.mark.skipif(not EMOTIONAL_PALETTE_AVAILABLE, reason="EmotionalPalette module not available")
    def test_analyze_sentiment_positive(self):
        """Test sentiment analysis for positive text."""
        palette = EmotionalPalette()
        
        scores = palette.analyze_sentiment("I am so happy and joyful today!")
        
        assert scores["Joy"] > 0
    
    @pytest.mark.skipif(not EMOTIONAL_PALETTE_AVAILABLE, reason="EmotionalPalette module not available")
    def test_analyze_sentiment_negative(self):
        """Test sentiment analysis for negative text."""
        palette = EmotionalPalette()
        
        scores = palette.analyze_sentiment("I feel so sad and afraid")
        
        assert scores["Sadness"] > 0 or scores["Fear"] > 0
    
    @pytest.mark.skipif(not EMOTIONAL_PALETTE_AVAILABLE, reason="EmotionalPalette module not available")
    def test_analyze_sentiment_korean(self):
        """Test sentiment analysis with Korean text."""
        palette = EmotionalPalette()
        
        scores = palette.analyze_sentiment("오늘 정말 행복해요 기쁨이 넘쳐요")
        
        assert scores["Joy"] > 0
    
    @pytest.mark.skipif(not EMOTIONAL_PALETTE_AVAILABLE, reason="EmotionalPalette module not available")
    def test_analyze_sentiment_neutral(self):
        """Test sentiment analysis for neutral text."""
        palette = EmotionalPalette()
        
        scores = palette.analyze_sentiment("The weather is normal")
        
        # Should default to mild trust
        assert scores["Trust"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
