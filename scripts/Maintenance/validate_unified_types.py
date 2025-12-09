#!/usr/bin/env python3
"""
Quick validation script for unified_types.py
Tests the consolidated Experience and EmotionalState classes
"""

import sys
sys.path.insert(0, '/home/runner/work/Elysia/Elysia')

from datetime import datetime
from Core.Memory.unified_types import (
    Experience, EmotionalState, EmotionalStateFactory,
    migrate_experience_from_old, migrate_emotional_state_from_old
)

def test_experience_basic():
    """Test basic Experience creation"""
    print("✓ Testing basic Experience creation...")
    exp = Experience(content="Test experience", type="episode")
    assert exp.content == "Test experience"
    assert exp.type == "episode"
    assert exp.layer == "soul"
    print("  ✅ Basic Experience works!")

def test_experience_from_conversation():
    """Test conversation-style creation"""
    print("✓ Testing Experience.from_conversation()...")
    exp = Experience.from_conversation(
        content="Hello, how are you?",
        intensity=0.7,
        context="Greeting"
    )
    assert exp.type == "conversation"
    assert exp.content == "Hello, how are you?"
    print("  ✅ from_conversation() works!")

def test_experience_from_learning():
    """Test learning-style creation"""
    print("✓ Testing Experience.from_learning()...")
    exp = Experience.from_learning(
        context={"input": "question"},
        action={"type": "respond"},
        outcome={"success": True},
        feedback=0.8,
        layer="soul"
    )
    assert exp.type == "learning"
    assert exp.feedback == 0.8
    print("  ✅ from_learning() works!")

def test_experience_from_divine():
    """Test divine-style creation"""
    print("✓ Testing Experience.from_divine()...")
    exp = Experience.from_divine(
        truth=0.9,
        emotion=0.5,
        causality=0.6,
        beauty=0.7
    )
    assert exp.type == "divine"
    assert exp.truth == 0.9
    print("  ✅ from_divine() works!")

def test_experience_serialization():
    """Test Experience serialization"""
    print("✓ Testing Experience serialization...")
    exp = Experience(
        content="Test",
        type="episode",
        tags=["important", "memory"]
    )
    
    data = exp.to_dict()
    assert isinstance(data, dict)
    assert data['content'] == "Test"
    
    exp2 = Experience.from_dict(data)
    assert exp2.content == exp.content
    print("  ✅ Serialization works!")

def test_emotional_state_presets():
    """Test EmotionalState presets"""
    print("✓ Testing EmotionalState presets...")
    
    neutral = EmotionalState.neutral()
    assert neutral.name == "neutral"
    
    calm = EmotionalState.calm()
    assert calm.name == "calm"
    
    hopeful = EmotionalState.hopeful()
    assert hopeful.name == "hopeful"
    assert "joy" in hopeful.secondary_emotions
    
    passionate = EmotionalState.passionate()
    assert passionate.source_spirit == "Creativity"
    
    melancholy = EmotionalState.melancholy()
    assert melancholy.source_spirit == "Memory"
    
    print("  ✅ All presets work!")

def test_emotional_state_factory():
    """Test EmotionalStateFactory"""
    print("✓ Testing EmotionalStateFactory...")
    
    state = EmotionalStateFactory.get("calm")
    assert state.name == "calm"
    
    state = EmotionalStateFactory.get("unknown")
    assert state.name == "neutral"  # Fallback
    
    # Test creating from spirit energy
    state = EmotionalStateFactory.create_from_spirit("Creativity", 0.8)
    assert state.source_spirit == "Creativity"
    assert state.intensity == 0.8
    # Fire spirit should have positive temperature
    assert state.temperature > 0 or state.name in ["passion", "passionate"]
    
    print("  ✅ Factory works!")

def test_integration():
    """Test integrated usage"""
    print("✓ Testing integrated usage...")
    
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
    
    print("  ✅ Integration works!")

def test_migration():
    """Test migration from old classes"""
    print("✓ Testing migration...")
    
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
    
    print("  ✅ Migration works!")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Testing Unified Types (Experience + EmotionalState)")
    print("="*60 + "\n")
    
    tests = [
        test_experience_basic,
        test_experience_from_conversation,
        test_experience_from_learning,
        test_experience_from_divine,
        test_experience_serialization,
        test_emotional_state_presets,
        test_emotional_state_factory,
        test_integration,
        test_migration,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            failed.append((test.__name__, str(e)))
            print(f"  ❌ FAILED: {e}")
    
    print("\n" + "="*60)
    if failed:
        print(f"❌ {len(failed)} test(s) failed:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        sys.exit(1)
    else:
        print("✅ All tests passed!")
        print("="*60 + "\n")
        print("🎉 Unified types are working correctly!")
        print("📦 Ready for P1.1 consolidation!")
        sys.exit(0)

if __name__ == "__main__":
    main()
