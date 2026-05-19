"""
Tests for ConversationMemory system
"""

import sys
from pathlib import Path
import tempfile
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Core.Memory.conversation_memory import (
    ConversationMemory,
    ConversationTurn,
    UserProfile,
    create_conversation_memory
)


def test_basic_turn_management():
    """Test adding and retrieving turns"""
    print("Test 1: Basic Turn Management")
    print("-" * 60)
    
    memory = create_conversation_memory(context_turns=3)
    
    # Add turns
    memory.add_turn("Hello", "Hi there!", language="en")
    memory.add_turn("How are you?", "I'm doing well!", language="en")
    memory.add_turn("Tell me a story", "Once upon a time...", language="en")
    
    context = memory.get_context()
    assert len(context) == 3, f"Expected 3 turns, got {len(context)}"
    
    # Add one more (should overflow sliding window)
    memory.add_turn("Continue", "And then...", language="en")
    context = memory.get_context()
    assert len(context) == 3, f"Expected 3 turns (max), got {len(context)}"
    
    # Check oldest was removed
    assert context[0].user_message == "How are you?", "Sliding window not working correctly"
    
    print(f"✓ Sliding window works: {len(context)} turns in context")
    print(f"✓ Total turns tracked: {memory.turn_counter}")
    print()


def test_context_formatting():
    """Test different context format styles"""
    print("Test 2: Context Formatting")
    print("-" * 60)
    
    memory = create_conversation_memory()
    
    memory.add_turn(
        "안녕하세요",
        "안녕하세요! 반가워요.",
        assistant_emotion="hopeful",
        topics=["greeting"],
        language="ko"
    )
    memory.add_turn(
        "기분이 어때요?",
        "기쁘고 설레요!",
        assistant_emotion="joyful",
        topics=["emotion"],
        language="ko"
    )
    
    # Test simple format
    simple = memory.get_context_string(format_style="simple")
    assert "User:" in simple and "Assistant:" in simple
    print("Simple format:")
    print(simple)
    print()
    
    # Test emotional format
    emotional = memory.get_context_string(format_style="emotional")
    assert "(hopeful)" in emotional or "(joyful)" in emotional
    print("Emotional format:")
    print(emotional)
    print()
    
    print("✓ Context formatting works")
    print()


def test_emotional_tracking():
    """Test emotional arc tracking"""
    print("Test 3: Emotional Arc Tracking")
    print("-" * 60)
    
    memory = create_conversation_memory()
    
    # Simulate emotional conversation
    emotions = [
        (0.5, "hopeful"),
        (0.7, "joyful"),
        (0.3, "calm"),
        (-0.2, "thoughtful"),
    ]
    
    for valence, emotion in emotions:
        memory.add_turn(
            f"Turn with valence {valence}",
            "Response",
            assistant_emotion=emotion,
            emotional_valence=valence
        )
    
    arc = memory.get_emotional_arc()
    assert len(arc) == 4, f"Expected 4 valence values, got {len(arc)}"
    assert arc[0] == 0.5, "First valence incorrect"
    assert arc[-1] == -0.2, "Last valence incorrect"
    
    print(f"Emotional arc: {arc}")
    print("✓ Emotional tracking works")
    print()


def test_topic_tracking():
    """Test topic analysis"""
    print("Test 4: Topic Tracking")
    print("-" * 60)
    
    memory = create_conversation_memory()
    
    # Add turns with various topics
    memory.add_turn("Let's talk about AI", "Sure!", topics=["AI", "technology"])
    memory.add_turn("Machine learning is fascinating", "Indeed!", topics=["AI", "machine_learning"])
    memory.add_turn("What about poetry?", "I love poetry!", topics=["poetry", "art"])
    memory.add_turn("AI poetry", "Interesting combo!", topics=["AI", "poetry"])
    
    topics = memory.get_dominant_topics(top_k=3)
    print(f"Dominant topics: {topics}")
    
    # AI should be most common
    assert topics[0][0] == "AI", f"Expected 'AI' as top topic, got '{topics[0][0]}'"
    assert topics[0][1] == 3, f"Expected AI count=3, got {topics[0][1]}"
    
    print("✓ Topic tracking works")
    print()


def test_user_profile_learning():
    """Test user profile learning"""
    print("Test 5: User Profile Learning")
    print("-" * 60)
    
    memory = create_conversation_memory()
    
    # Simulate user interactions
    for i in range(5):
        memory.add_turn(
            f"Message {i+1} in English",
            "Response",
            topics=["chat", "test"],
            language="en",
            user_id="test_user"
        )
    
    profile = memory.get_user_profile("test_user")
    assert profile is not None, "Profile not created"
    assert profile.user_id == "test_user"
    assert profile.preferred_language == "en", f"Expected 'en', got '{profile.preferred_language}'"
    assert profile.total_turns == 5, f"Expected 5 turns, got {profile.total_turns}"
    assert "chat" in profile.favorite_topics or "test" in profile.favorite_topics, "Topics not learned"
    
    print(f"✓ Profile created for user: {profile.user_id}")
    print(f"  Language: {profile.preferred_language}")
    print(f"  Total turns: {profile.total_turns}")
    print(f"  Favorite topics: {profile.favorite_topics}")
    print()


def test_session_management():
    """Test session archiving"""
    print("Test 6: Session Management")
    print("-" * 60)
    
    memory = create_conversation_memory(context_turns=5)
    memory.max_session_turns = 10  # Set small for testing
    
    # Add turns to trigger archiving
    for i in range(12):
        memory.add_turn(f"User {i}", f"Response {i}")
    
    assert memory.session_counter >= 1, "Session not archived"
    assert len(memory.session_turns) < 10, "Session not cleared after archive"
    
    print(f"✓ Session archived automatically")
    print(f"  Sessions: {memory.session_counter}")
    print(f"  Current session turns: {len(memory.session_turns)}")
    print()


def test_persistence():
    """Test save/load to file"""
    print("Test 7: Persistence (Save/Load)")
    print("-" * 60)
    
    # Create and populate memory
    memory1 = create_conversation_memory()
    memory1.add_turn("Hello", "Hi!", user_id="user1", language="en", topics=["greeting"])
    memory1.add_turn("Goodbye", "Bye!", user_id="user1", language="en", topics=["farewell"])
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        memory1.save_to_file(temp_path)
        
        # Load into new instance
        memory2 = create_conversation_memory()
        memory2.load_from_file(temp_path)
        
        # Verify
        assert len(memory2.context) == 2, f"Expected 2 turns, got {len(memory2.context)}"
        assert memory2.turn_counter == memory1.turn_counter, "Turn counter not restored"
        assert "user1" in memory2.user_profiles, "User profile not restored"
        
        context = memory2.get_context()
        assert context[0].user_message == "Hello", "Turn content not restored"
        
        print(f"✓ Save/load works")
        print(f"  Turns restored: {len(memory2.context)}")
        print(f"  Profiles restored: {len(memory2.user_profiles)}")
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print()


def test_multilingual_support():
    """Test multilingual conversation tracking"""
    print("Test 8: Multilingual Support")
    print("-" * 60)
    
    memory = create_conversation_memory()
    
    # Different languages
    memory.add_turn("Hello", "Hi!", language="en")
    memory.add_turn("안녕하세요", "안녕!", language="ko")
    memory.add_turn("こんにちは", "こんにちは!", language="ja")
    
    context = memory.get_context()
    languages = [turn.language for turn in context]
    
    assert "en" in languages, "English not tracked"
    assert "ko" in languages, "Korean not tracked"
    assert "ja" in languages, "Japanese not tracked"
    
    print(f"✓ Multilingual tracking works")
    print(f"  Languages: {set(languages)}")
    print()


def test_integration_ready():
    """Test that API is ready for integration"""
    print("Test 9: Integration API Check")
    print("-" * 60)
    
    memory = create_conversation_memory()
    
    # Verify key methods exist and work
    assert hasattr(memory, 'add_turn'), "Missing add_turn method"
    assert hasattr(memory, 'get_context'), "Missing get_context method"
    assert hasattr(memory, 'get_context_string'), "Missing get_context_string method"
    assert hasattr(memory, 'get_emotional_arc'), "Missing get_emotional_arc method"
    assert hasattr(memory, 'get_user_profile'), "Missing get_user_profile method"
    assert hasattr(memory, 'save_to_file'), "Missing save_to_file method"
    assert hasattr(memory, 'load_from_file'), "Missing load_from_file method"
    
    # Test convenience function
    mem2 = create_conversation_memory(context_turns=20)
    assert mem2.max_context_turns == 20, "Convenience function not working"
    
    print("✓ All integration APIs present")
    print("✓ Ready for EmotionalEngine integration")
    print()


def run_all_tests():
    """Run all conversation memory tests"""
    print("\n" + "=" * 70)
    print("=" * 70)
    print(" " * 15 + "CONVERSATION MEMORY TESTS")
    print("=" * 70)
    print("=" * 70)
    print()
    
    try:
        test_basic_turn_management()
        test_context_formatting()
        test_emotional_tracking()
        test_topic_tracking()
        test_user_profile_learning()
        test_session_management()
        test_persistence()
        test_multilingual_support()
        test_integration_ready()
        
        print("=" * 70)
        print("✅ ALL TESTS PASSED (9/9)")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ Turn management and sliding window")
        print("  ✓ Context formatting (simple, detailed, emotional)")
        print("  ✓ Emotional arc tracking")
        print("  ✓ Topic analysis")
        print("  ✓ User profile learning")
        print("  ✓ Session management")
        print("  ✓ Persistence (save/load)")
        print("  ✓ Multilingual support")
        print("  ✓ Integration-ready API")
        print()
        print("🎯 Ready for Phase 7 integration!")
        print()
        return True
        
    except Exception as e:
        print("=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
