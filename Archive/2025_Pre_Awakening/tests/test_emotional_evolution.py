"""
Tests for Emotional Evolution System

Tests the ability of Elysia's emotions to learn and evolve from experiences.
"""

import pytest
import os
import tempfile
from Core.FoundationLayer.Foundation.emotional_evolution import (
    EmotionalEvolutionEngine,
    EmotionalExperience,
    EmotionalPattern,
    EmotionalTrigger,
    create_emotional_evolution_engine
)


def test_emotional_evolution_initialization():
    """Test basic initialization"""
    engine = create_emotional_evolution_engine()
    
    assert engine is not None
    assert engine.total_experiences_count == 0
    assert engine.emotional_maturity == 0.0
    assert len(engine.experiences) == 0
    assert len(engine.patterns) == 0
    assert len(engine.triggers) == 0


def test_record_single_experience():
    """Test recording a single emotional experience"""
    engine = create_emotional_evolution_engine()
    
    exp = engine.record_experience(
        primary_emotion="joy",
        valence=0.8,
        arousal=0.7,
        intensity=0.9,
        trigger="received praise",
        context="work achievement",
        topics=["work", "achievement"],
        outcome="positive"
    )
    
    assert exp is not None
    assert exp.primary_emotion == "joy"
    assert exp.valence == 0.8
    assert exp.arousal == 0.7
    assert exp.intensity == 0.9
    assert exp.trigger == "received praise"
    
    assert engine.total_experiences_count == 1
    assert len(engine.experiences) == 1


def test_pattern_learning():
    """Test that patterns are learned from repeated experiences"""
    engine = create_emotional_evolution_engine(pattern_learning_threshold=3)
    
    # Record same trigger multiple times
    for i in range(5):
        engine.record_experience(
            primary_emotion="joy",
            valence=0.7 + i * 0.02,  # Slightly varying
            arousal=0.6,
            intensity=0.8,
            trigger="morning coffee",
            context="daily routine",
            outcome="positive"
        )
    
    # Check pattern formed
    assert "morning coffee" in engine.patterns
    pattern = engine.patterns["morning coffee"]
    
    assert pattern.occurrence_count == 5
    assert pattern.average_valence > 0.7
    assert pattern.average_arousal == 0.6
    assert "joy" in pattern.primary_emotions
    assert pattern.positive_outcomes == 5
    assert pattern.maturity_level > 0.0


def test_emotion_prediction():
    """Test predicting emotional reaction from learned patterns"""
    engine = create_emotional_evolution_engine(pattern_learning_threshold=2)
    
    # Learn pattern
    for _ in range(3):
        engine.record_experience(
            primary_emotion="sadness",
            valence=-0.6,
            arousal=0.4,
            intensity=0.7,
            trigger="rainy day",
            context="weather",
            outcome="neutral"
        )
    
    # Predict reaction
    prediction = engine.predict_reaction("rainy day")
    
    assert prediction is not None
    assert prediction['valence'] < 0  # Should predict negative
    assert prediction['primary_emotion'] == "sadness"
    assert prediction['confidence'] > 0


def test_joy_trigger_formation():
    """Test formation of joy triggers from intense positive experiences"""
    engine = create_emotional_evolution_engine(trigger_formation_threshold=0.8)
    
    # Record intense joyful experience
    engine.record_experience(
        primary_emotion="joy",
        valence=0.9,
        arousal=0.9,
        intensity=0.95,  # Very intense
        trigger="surprise birthday party",
        context="celebration",
        outcome="positive"
    )
    
    # Check if joy trigger formed
    joy_triggers = [k for k in engine.triggers.keys() if k.startswith('joy:')]
    assert len(joy_triggers) > 0
    
    # Check trigger properties
    trigger_key = joy_triggers[0]
    trigger = engine.triggers[trigger_key]
    assert trigger.trigger_type == "joy"
    assert trigger.learned_valence > 0.8
    assert trigger.strength == 1.0


def test_trauma_trigger_formation():
    """Test formation of trauma triggers from intense negative experiences"""
    engine = create_emotional_evolution_engine(trigger_formation_threshold=0.8)
    
    # Record intense traumatic experience
    engine.record_experience(
        primary_emotion="fear",
        valence=-0.9,
        arousal=0.9,
        intensity=0.95,  # Very intense
        trigger="loud sudden noise accident",
        context="frightening event",
        outcome="negative"
    )
    
    # Check if trauma trigger formed
    trauma_triggers = [k for k in engine.triggers.keys() if k.startswith('trauma:')]
    assert len(trauma_triggers) > 0
    
    # Check trigger properties
    trigger_key = trauma_triggers[0]
    trigger = engine.triggers[trigger_key]
    assert trigger.trigger_type == "trauma"
    assert trigger.learned_valence < -0.8
    assert trigger.strength == 1.0


def test_trigger_reinforcement():
    """Test that triggers strengthen with repeated exposure"""
    engine = create_emotional_evolution_engine(trigger_formation_threshold=0.8)
    
    # Record intense joyful experience twice
    for _ in range(2):
        engine.record_experience(
            primary_emotion="joy",
            valence=0.9,
            arousal=0.9,
            intensity=0.95,
            trigger="chocolate cake",
            context="dessert",
            outcome="positive"
        )
    
    # Find chocolate trigger
    joy_triggers = [k for k in engine.triggers.keys() if 'chocolate' in k]
    assert len(joy_triggers) > 0
    
    trigger = engine.triggers[joy_triggers[0]]
    assert trigger.reinforcement_count == 2
    assert trigger.strength >= 1.0  # Should be strengthened


def test_emotional_maturity_growth():
    """Test that emotional maturity increases with experiences"""
    engine = create_emotional_evolution_engine()
    
    initial_maturity = engine.emotional_maturity
    
    # Record various experiences
    emotions = ["joy", "sadness", "anger", "fear"]
    for i in range(20):
        engine.record_experience(
            primary_emotion=emotions[i % len(emotions)],
            valence=(-1 if i % 2 else 1) * 0.5,
            arousal=0.6,
            intensity=0.7,
            trigger=f"event_{i % 5}",  # Repeated triggers
            context="various",
            outcome="neutral"
        )
    
    # Maturity should increase
    assert engine.emotional_maturity > initial_maturity
    assert engine.emotional_maturity > 0.0


def test_emotional_growth_report():
    """Test generation of emotional growth report"""
    engine = create_emotional_evolution_engine()
    
    # Record some experiences
    for i in range(10):
        engine.record_experience(
            primary_emotion="joy" if i < 5 else "sadness",
            valence=0.7 if i < 5 else -0.6,
            arousal=0.6,
            intensity=0.9 if i == 0 else 0.7,  # First one intense
            trigger=f"trigger_{i % 3}",
            context="test",
            outcome="positive" if i < 5 else "negative"
        )
    
    report = engine.get_emotional_growth_report()
    
    assert report['total_experiences'] == 10
    assert report['patterns_learned'] >= 3
    assert 'emotional_maturity' in report
    assert 'most_common_emotions' in report
    assert 'emotional_stability' in report
    assert len(report['most_common_emotions']) > 0


def test_multilingual_support():
    """Test multilingual experience recording"""
    engine = create_emotional_evolution_engine()
    
    # Korean
    engine.set_language("ko")
    exp_ko = engine.record_experience(
        primary_emotion="기쁨",
        valence=0.8,
        arousal=0.7,
        intensity=0.8,
        trigger="칭찬",
        context="업무",
        outcome="positive"
    )
    assert exp_ko.language == "ko"
    
    # English
    engine.set_language("en")
    exp_en = engine.record_experience(
        primary_emotion="joy",
        valence=0.8,
        arousal=0.7,
        intensity=0.8,
        trigger="praise",
        context="work",
        outcome="positive"
    )
    assert exp_en.language == "en"
    
    # Japanese
    engine.set_language("ja")
    exp_ja = engine.record_experience(
        primary_emotion="喜び",
        valence=0.8,
        arousal=0.7,
        intensity=0.8,
        trigger="褒め言葉",
        context="仕事",
        outcome="positive"
    )
    assert exp_ja.language == "ja"


def test_persistence():
    """Test saving and loading emotional evolution state"""
    engine = create_emotional_evolution_engine()
    
    # Record experiences
    for i in range(5):
        engine.record_experience(
            primary_emotion="joy",
            valence=0.7,
            arousal=0.6,
            intensity=0.95 if i == 0 else 0.7,  # First one forms trigger
            trigger="happy music",
            context="music",
            outcome="positive"
        )
    
    original_count = engine.total_experiences_count
    original_maturity = engine.emotional_maturity
    original_patterns = len(engine.patterns)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        engine.save_to_file(temp_path)
        
        # Create new engine and load
        new_engine = create_emotional_evolution_engine()
        new_engine.load_from_file(temp_path)
        
        # Verify loaded state
        assert new_engine.total_experiences_count == original_count
        assert new_engine.emotional_maturity == original_maturity
        assert len(new_engine.patterns) == original_patterns
        assert len(new_engine.experiences) == original_count
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_trigger_based_prediction():
    """Test prediction based on triggered words"""
    engine = create_emotional_evolution_engine(trigger_formation_threshold=0.8)
    
    # Form a trauma trigger
    engine.record_experience(
        primary_emotion="fear",
        valence=-0.9,
        arousal=0.95,
        intensity=0.95,
        trigger="spider encounter",
        context="phobia",
        outcome="negative"
    )
    
    # Predict reaction to word "spider"
    prediction = engine.predict_reaction("I saw a spider")
    
    assert prediction is not None
    assert prediction['source'] == 'trigger'
    assert prediction['valence'] < 0  # Should predict negative
    assert prediction['intensity'] > 0


def test_emotional_stability_calculation():
    """Test emotional stability calculation"""
    engine = create_emotional_evolution_engine()
    
    # Record very stable emotions (all similar)
    for i in range(50):
        engine.record_experience(
            primary_emotion="calm",
            valence=0.5,  # All the same
            arousal=0.3,  # All the same
            intensity=0.5,
            trigger=f"routine_{i}",
            context="daily",
            outcome="neutral"
        )
    
    report = engine.get_emotional_growth_report()
    stability = report['emotional_stability']
    
    # High stability due to consistent emotions
    assert stability > 0.7
    
    # Now add volatile emotions
    for i in range(20):
        engine.record_experience(
            primary_emotion="varying",
            valence=(-1 if i % 2 else 1) * 0.8,  # Wildly varying
            arousal=(i % 10) / 10.0,  # Varying
            intensity=0.8,
            trigger=f"volatile_{i}",
            context="chaotic",
            outcome="mixed"
        )
    
    new_report = engine.get_emotional_growth_report()
    new_stability = new_report['emotional_stability']
    
    # Stability should decrease
    assert new_stability < stability


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
