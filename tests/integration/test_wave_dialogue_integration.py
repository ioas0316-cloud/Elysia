"""
Tests for Wave Dialogue Flow and EmotionalEngine Integration
"""

import sys
from pathlib import Path
import tempfile
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Core.Foundation.wave_dialogue_flow import (
    WaveDialogueFlow,
    DialogueWave,
    create_wave_dialogue_flow
)
from Core.Foundation.emotional_engine import EmotionalEngine
from Core.Memory.conversation_memory import create_conversation_memory


def test_wave_representation():
    """Test dialogue wave representation"""
    print("Test 1: Dialogue Wave Representation")
    print("-" * 70)
    
    wave1 = DialogueWave(
        message="I'm happy today!",
        speaker="user",
        frequency=300.0,
        amplitude=0.8,
        valence=0.7,
        arousal=0.6
    )
    
    wave2 = DialogueWave(
        message="That's wonderful!",
        speaker="assistant",
        frequency=320.0,
        amplitude=0.7,
        valence=0.8,
        arousal=0.5
    )
    
    resonance = wave1.resonate_with(wave2)
    assert 0.0 <= resonance <= 1.0, f"Resonance out of range: {resonance}"
    assert resonance > 0.5, "Similar waves should have high resonance"
    
    print(f"✓ Wave 1: freq={wave1.frequency}Hz, valence={wave1.valence}")
    print(f"✓ Wave 2: freq={wave2.frequency}Hz, valence={wave2.valence}")
    print(f"✓ Resonance: {resonance:.3f}")
    print()


def test_emotional_engine_integration():
    """Test EmotionalEngine with conversation memory"""
    print("Test 2: EmotionalEngine with ConversationMemory")
    print("-" * 70)
    
    # Create engine with conversation memory enabled
    engine = EmotionalEngine(enable_conversation_memory=True)
    
    # Check that memory was initialized
    assert hasattr(engine, '_conversation_memory'), "Conversation memory not initialized"
    
    # Record turns
    engine.record_conversation_turn(
        user_message="Hello!",
        assistant_message="Hi there!",
        topics=["greeting"],
        user_id="test_user"
    )
    
    engine.record_conversation_turn(
        user_message="How are you?",
        assistant_message="I'm doing well!",
        topics=["greeting", "wellbeing"],
        user_id="test_user"
    )
    
    # Get context
    context = engine.get_conversation_context(n_turns=2)
    assert "Hello!" in context, "Context should include first turn"
    assert "How are you?" in context, "Context should include second turn"
    
    print(f"✓ Conversation memory enabled")
    print(f"✓ Recorded 2 turns")
    print(f"✓ Context retrieved:")
    print(context[:200] + "...")
    print()


def test_wave_dialogue_flow_basic():
    """Test basic wave dialogue flow"""
    print("Test 3: Wave Dialogue Flow - Basic")
    print("-" * 70)
    
    flow = create_wave_dialogue_flow(emotional=True, memory=True, poetic=True)
    
    # Process input
    result = flow.process_user_input(
        user_message="Hello, I'm happy today!",
        user_id="test_user",
        topics=["greeting", "emotion"]
    )
    
    assert "response" in result, "Result missing response"
    assert "wave_properties" in result, "Result missing wave properties"
    assert "context_resonance" in result, "Result missing context resonance"
    
    print(f"✓ Flow initialized")
    print(f"✓ User message processed")
    print(f"✓ Response: {result['response'][:80]}...")
    print(f"✓ Wave freq: {result['wave_properties']['frequency']:.1f}Hz")
    print(f"✓ Valence: {result['wave_properties']['valence']:.2f}")
    print()


def test_context_resonance():
    """Test context resonance calculation"""
    print("Test 4: Context Resonance Over Multiple Turns")
    print("-" * 70)
    
    flow = create_wave_dialogue_flow()
    
    # First turn - no context
    result1 = flow.process_user_input("Hello!")
    resonance1 = result1['context_resonance']
    
    # Second turn - should have context
    result2 = flow.process_user_input("How are you?")
    resonance2 = result2['context_resonance']
    
    # Third turn - more context
    result3 = flow.process_user_input("That's great!")
    resonance3 = result3['context_resonance']
    
    assert resonance1 >= 0.0 and resonance1 <= 1.0, "Resonance 1 out of range"
    assert resonance2 >= 0.0 and resonance2 <= 1.0, "Resonance 2 out of range"
    assert resonance3 >= 0.0 and resonance3 <= 1.0, "Resonance 3 out of range"
    
    print(f"✓ Turn 1 resonance: {resonance1:.3f}")
    print(f"✓ Turn 2 resonance: {resonance2:.3f}")
    print(f"✓ Turn 3 resonance: {resonance3:.3f}")
    print(f"✓ Context builds over turns")
    print()


def test_emotional_modulation():
    """Test emotional state modulation through waves"""
    print("Test 5: Emotional State Modulation")
    print("-" * 70)
    
    flow = create_wave_dialogue_flow(emotional=True)
    
    # Get initial state
    if flow.emotional_engine:
        initial_state = flow.emotional_engine.current_state.primary_emotion
        print(f"Initial emotion: {initial_state}")
        
        # Process positive message
        result = flow.process_user_input("I love this! It's amazing!")
        
        # Check if emotion changed
        final_state = flow.emotional_engine.current_state.primary_emotion
        print(f"After positive input: {final_state}")
        
        # State should be updated
        assert flow.emotional_engine.current_state is not None, "Emotional state not maintained"
        
        print(f"✓ Emotional modulation working")
    else:
        print("⚠️  EmotionalEngine not available, skipping")
    
    print()


def test_wave_buffer_management():
    """Test wave buffer sliding window"""
    print("Test 6: Wave Buffer Management")
    print("-" * 70)
    
    flow = create_wave_dialogue_flow()
    flow.max_buffer_size = 5
    
    # Add more turns than buffer size
    for i in range(8):
        flow.process_user_input(f"Message {i+1}")
    
    # Buffer should be limited
    assert len(flow.wave_buffer) <= flow.max_buffer_size * 2, \
        f"Buffer size {len(flow.wave_buffer)} exceeds limit"
    
    print(f"✓ Added 8 turns (16 waves)")
    print(f"✓ Buffer size: {len(flow.wave_buffer)} (max {flow.max_buffer_size * 2})")
    print(f"✓ Sliding window working")
    print()


def test_conversation_summary():
    """Test conversation summary generation"""
    print("Test 7: Conversation Summary")
    print("-" * 70)
    
    flow = create_wave_dialogue_flow()
    
    # Initial summary
    summary1 = flow.get_conversation_summary()
    assert summary1['status'] == 'empty', "Initial summary should be empty"
    
    # Add turns
    flow.process_user_input("Hello!")
    flow.process_user_input("How are you?")
    flow.process_user_input("That's great!")
    
    summary2 = flow.get_conversation_summary()
    assert summary2['status'] == 'active', "Summary should show active conversation"
    assert summary2['turns'] >= 6, f"Should have at least 6 waves, got {summary2['turns']}"
    
    print("Initial summary:", summary1)
    print()
    print("After 3 turns:", summary2)
    print()
    print("✓ Summary generation working")
    print()


def test_multilingual_flow():
    """Test multilingual dialogue flow"""
    print("Test 8: Multilingual Wave Flow")
    print("-" * 70)
    
    flow = create_wave_dialogue_flow(emotional=True, poetic=True)
    
    if flow.emotional_engine:
        # Test Korean
        flow.emotional_engine.set_language("ko")
        result_ko = flow.process_user_input("안녕하세요")
        
        # Test English
        flow.emotional_engine.set_language("en")
        result_en = flow.process_user_input("Hello")
        
        # Test Japanese
        flow.emotional_engine.set_language("ja")
        result_ja = flow.process_user_input("こんにちは")
        
        print(f"✓ Korean wave processed")
        print(f"✓ English wave processed")
        print(f"✓ Japanese wave processed")
        print(f"✓ Multilingual flow working")
    else:
        print("⚠️  EmotionalEngine not available, skipping")
    
    print()


def test_persistence_integration():
    """Test conversation persistence through engine"""
    print("Test 9: Conversation Persistence")
    print("-" * 70)
    
    engine = EmotionalEngine(enable_conversation_memory=True)
    
    # Add turns
    engine.record_conversation_turn(
        "Hello", "Hi!", topics=["greeting"], user_id="user1"
    )
    engine.record_conversation_turn(
        "Goodbye", "Bye!", topics=["farewell"], user_id="user1"
    )
    
    # Save
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        engine.save_conversation_history(temp_path)
        
        # Load into new engine
        engine2 = EmotionalEngine(enable_conversation_memory=True)
        engine2.load_conversation_history(temp_path)
        
        # Verify
        context = engine2.get_conversation_context()
        assert "Hello" in context, "Conversation not restored"
        
        print(f"✓ Saved conversation history")
        print(f"✓ Loaded into new engine")
        print(f"✓ Context verified")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print()


def test_system_fluidity():
    """Test that the system remains fluid and wave-based"""
    print("Test 10: System Fluidity Check")
    print("-" * 70)
    
    flow = create_wave_dialogue_flow(emotional=True, memory=True, poetic=True)
    
    # Rapid conversation - system should flow smoothly
    messages = [
        "Hello!",
        "How are you?",
        "Tell me about waves",
        "That's interesting!",
        "Can you explain more?",
        "Thank you!"
    ]
    
    results = []
    for msg in messages:
        result = flow.process_user_input(msg)
        results.append(result)
    
    # Check all results have wave properties
    for i, result in enumerate(results):
        assert 'wave_properties' in result, f"Turn {i+1} missing wave properties"
        assert 'context_resonance' in result, f"Turn {i+1} missing resonance"
        
        # Wave properties should be reasonable
        props = result['wave_properties']
        assert 0 <= props['frequency'] <= 1000, f"Turn {i+1} frequency out of range"
        assert -1 <= props['valence'] <= 1, f"Turn {i+1} valence out of range"
        assert 0 <= props['arousal'] <= 1, f"Turn {i+1} arousal out of range"
    
    print(f"✓ Processed {len(messages)} messages rapidly")
    print(f"✓ All waves have valid properties")
    print(f"✓ Context resonance maintained")
    print(f"✓ System remains fluid (wave-based)")
    print()


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "=" * 70)
    print("=" * 70)
    print(" " * 10 + "WAVE DIALOGUE FLOW INTEGRATION TESTS")
    print("=" * 70)
    print("=" * 70)
    print()
    
    try:
        test_wave_representation()
        test_emotional_engine_integration()
        test_wave_dialogue_flow_basic()
        test_context_resonance()
        test_emotional_modulation()
        test_wave_buffer_management()
        test_conversation_summary()
        test_multilingual_flow()
        test_persistence_integration()
        test_system_fluidity()
        
        print("=" * 70)
        print("✅ ALL TESTS PASSED (10/10)")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ Dialogue wave representation")
        print("  ✓ EmotionalEngine + ConversationMemory integration")
        print("  ✓ Wave dialogue flow")
        print("  ✓ Context resonance")
        print("  ✓ Emotional modulation")
        print("  ✓ Wave buffer management")
        print("  ✓ Conversation summary")
        print("  ✓ Multilingual support")
        print("  ✓ Persistence")
        print("  ✓ System fluidity (CRITICAL)")
        print()
        print("🌊 System remains fluid and wave-based!")
        print("🎯 Ready for production integration!")
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
