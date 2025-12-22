"""
Tests for unified_types.py - Consolidated Experience and EmotionalState

Tests ensure:
1. Unified classes work correctly
2. All legacy use cases are supported
3. Migration from old classes works
4. Backward compatibility maintained
"""

import pytest
from datetime import datetime
from Core.Memory.unified_types import (
    Experience, EmotionalState, EmotionalStateFactory,
    migrate_experience_from_old, migrate_emotional_state_from_old
)


class TestExperience:
    """Test unified Experience class"""
    
    def test_basic_creation(self):
        """Test basic Experience creation"""
        exp = Experience(content="Test experience", type="episode")
        assert exp.content == "Test experience"
        assert exp.type == "episode"
        assert exp.layer == "soul"
        assert isinstance(exp.timestamp, float)
    
    def test_from_conversation(self):
        """Test conversation-style creation (experience_stream.py compatibility)"""
        exp = Experience.from_conversation(
            content="Hello, how are you?",
            intensity=0.7,
            context="Greeting"
        )
        assert exp.type == "conversation"
        assert exp.content == "Hello, how are you?"
        assert exp.intensity == 0.7
    
    def test_from_learning(self):
        """Test learning-style creation (experience_learner.py compatibility)"""
        exp = Experience.from_learning(
            context={"input": "question"},
            action={"type": "respond"},
            outcome={"success": True},
            feedback=0.8,
            layer="soul"
        )
        assert exp.type == "learning"
        assert exp.feedback == 0.8
        assert exp.action["type"] == "respond"
    
    def test_from_divine(self):
        """Test divine-style creation (divine_engine.py compatibility)"""
        exp = Experience.from_divine(
            truth=0.9,
            emotion=0.5,
            causality=0.6,
            beauty=0.7
        )
        assert exp.type == "divine"
        assert exp.truth == 0.9
        assert exp.causality == 0.6
        assert exp.beauty == 0.7
    
    def test_serialization(self):
        """Test to_dict and from_dict"""
        exp = Experience(
            content="Test",
            type="episode",
            tags=["important", "memory"]
        )
        
        data = exp.to_dict()
        assert isinstance(data, dict)
        assert data['content'] == "Test"
        assert "important" in data['tags']
        
        exp2 = Experience.from_dict(data)
        assert exp2.content == exp.content
        assert exp2.tags == exp.tags
    
    def test_with_emotional_state(self):
        """Test Experience with EmotionalState"""
        emotion = EmotionalState.hopeful()
        exp = Experience(
            content="Good news!",
            emotional_state=emotion
        )
        
        assert exp.emotional_state.name == "hopeful"
        assert exp.emotional_state.valence > 0


class TestEmotionalState:
    """Test unified EmotionalState class"""
    
    def test_basic_creation(self):
        """Test basic EmotionalState creation"""
        state = EmotionalState(
            name="test",
            valence=0.5,
            arousal=0.3,
            dominance=0.2
        )
        assert state.name == "test"
        assert state.valence == 0.5
        assert state.primary_emotion == "test"  # Auto-synced
    
    def test_presets(self):
        """Test preset emotional states"""
        neutral = EmotionalState.neutral()
        assert neutral.name == "neutral"
        assert neutral.valence == 0.0
        
        calm = EmotionalState.calm()
        assert calm.name == "calm"
        assert calm.valence > 0
        assert calm.arousal < 0.5
        
        hopeful = EmotionalState.hopeful()
        assert hopeful.name == "hopeful"
        assert hopeful.valence > 0.5
        assert "joy" in hopeful.secondary_emotions
        
        passionate = EmotionalState.passionate()
        assert passionate.source_spirit == "Creativity"
        assert passionate.temperature > 0.5
        
        melancholy = EmotionalState.melancholy()
        assert melancholy.source_spirit == "Memory"
        assert melancholy.valence < 0
    
    def test_factory(self):
        """Test EmotionalStateFactory"""
        state = EmotionalStateFactory.get("calm")
        assert state.name == "calm"
        
        state = EmotionalStateFactory.get("unknown")
        assert state.name == "neutral"  # Fallback
    
    def test_from_spirit(self):
        """Test creating from spirit energy"""
        state = EmotionalStateFactory.create_from_spirit("Creativity", 0.8)
        assert state.source_spirit == "Creativity"
        assert state.intensity == 0.8
        assert state.temperature > 0  # Fire spirit is hot
    
    def test_serialization(self):
        """Test to_dict and from_dict"""
        state = EmotionalState.hopeful()
        
        data = state.to_dict()
        assert isinstance(data, dict)
        assert data['name'] == "hopeful"
        
        state2 = EmotionalState.from_dict(data)
        assert state2.name == state.name
        assert state2.valence == state.valence


class TestMigration:
    """Test migration from old classes"""
    
    def test_migrate_experience_basic(self):
        """Test migrating a basic old Experience"""
        # Simulate old Experience object
        class OldExperience:
            def __init__(self):
                self.timestamp = datetime.now().timestamp()
                self.content = "Old experience"
                self.type = "episode"
                self.layer = "soul"
        
        old = OldExperience()
        new = migrate_experience_from_old(old)
        
        assert new.content == "Old experience"
        assert new.type == "episode"
        assert new.timestamp == old.timestamp
    
    def test_migrate_emotional_state_basic(self):
        """Test migrating a basic old EmotionalState"""
        # Simulate old EmotionalState object
        class OldEmotionalState:
            def __init__(self):
                self.name = "test"
                self.valence = 0.5
                self.arousal = 0.3
                self.dominance = 0.2
        
        old = OldEmotionalState()
        new = migrate_emotional_state_from_old(old)
        
        assert new.name == "test"
        assert new.valence == 0.5
        assert new.arousal == 0.3


class TestBackwardCompatibility:
    """Test backward compatibility with old code patterns"""
    
    def test_experience_stream_pattern(self):
        """Test pattern from experience_stream.py"""
        exp = Experience(
            timestamp=datetime.now().timestamp(),
            type="conversation",
            content="Hello",
            intensity=0.8,
            context={"general": "Greeting"}
        )
        
        # Should support to_dict() for logging
        data = exp.to_dict()
        assert data['type'] == "conversation"
        
        # Should support from_dict() for loading
        exp2 = Experience.from_dict(data)
        assert exp2.content == "Hello"
    
    def test_experience_learner_pattern(self):
        """Test pattern from experience_learner.py"""
        exp = Experience(
            timestamp=datetime.now().timestamp(),
            context={"input": "data"},
            action={"type": "process"},
            outcome={"result": "success"},
            feedback=0.9,
            layer="2D",
            tags=["learning", "positive"]
        )
        
        assert exp.feedback == 0.9
        assert "learning" in exp.tags
        assert exp.action["type"] == "process"
    
    def test_divine_engine_pattern(self):
        """Test pattern from divine_engine.py"""
        exp = Experience(
            truth=0.95,
            causality=0.7,
            beauty=0.8,
            meta={"branch": "main", "node_id": "123"}
        )
        
        assert exp.truth == 0.95
        assert exp.meta["branch"] == "main"
    
    def test_emotional_engine_pattern(self):
        """Test pattern from emotional_engine.py"""
        # Should support FEELING_PRESETS pattern via Factory
        presets = EmotionalStateFactory.PRESETS
        assert "neutral" in presets
        assert "calm" in presets
        assert "hopeful" in presets
        
        # Should support getting presets
        neutral = EmotionalStateFactory.get("neutral")
        assert neutral.valence == 0.0
    
    def test_spirit_emotion_pattern(self):
        """Test pattern from spirit_emotion.py"""
        # Should support creating from spirit energy
        creativity_emotion = EmotionalStateFactory.create_from_spirit("Creativity", 0.8)
        assert creativity_emotion.source_spirit == "Creativity"
        assert creativity_emotion.temperature > 0  # Fire is hot
        
        memory_emotion = EmotionalStateFactory.create_from_spirit("Memory", 0.6)
        assert memory_emotion.source_spirit == "Memory"
        assert memory_emotion.temperature < 0  # Water is cool


def test_integration():
    """Test integrated usage"""
    # Create emotion
    emotion = EmotionalState.passionate()
    
    # Create experience with emotion
    exp = Experience(
        content="Amazing discovery!",
        type="insight",
        emotional_state=emotion,
        truth=0.9,
        beauty=0.8,
        tags=["breakthrough", "creative"]
    )
    
    # Serialize
    data = exp.to_dict()
    
    # Deserialize
    exp2 = Experience.from_dict(data)
    
    # Verify
    assert exp2.content == "Amazing discovery!"
    assert exp2.emotional_state.name == "passion"
    assert exp2.truth == 0.9
    assert "breakthrough" in exp2.tags


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
