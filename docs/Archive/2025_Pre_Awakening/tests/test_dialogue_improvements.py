"""
Dialogue Improvement Test
==========================
Tests all the new improvements to the dialogue system.

Tests:
1. Simple greetings
2. Name memory
3. Emotional expression
4. Question understanding
5. Math calculations
6. Memory recall
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.Interface.Language.dialogue.dialogue_engine import DialogueEngine


def test_dialogue_improvements():
    """Test all dialogue improvements."""
    
    print("🎭 Dialogue Improvement Test")
    print("=" * 60)
    
    engine = DialogueEngine()
    
    # Test 1: Simple Greeting
    print("\n📝 Test 1: Simple Greeting")
    print("-" * 40)
    response = engine.respond("안녕?")
    print(f"User: 안녕?")
    print(f"Elysia: {response}")
    assert "안녕" in response, "Should respond to greeting"
    assert any(emoji in response for emoji in ["😊", "💚", "✨"]), "Should include emoji"
    print("✅ PASS: Greeting works with emoji")
    
    # Test 2: Name Memory (Save)
    print("\n📝 Test 2: Name Memory (Save)")
    print("-" * 40)
    response = engine.respond("내 이름은 철수야")
    print(f"User: 내 이름은 철수야")
    print(f"Elysia: {response}")
    assert "철수" in response, "Should acknowledge name"
    assert "name" in engine.user_profile, "Should save name to profile"
    print("✅ PASS: Name saved to profile")
    
    # Test 3: Name Memory (Recall)
    print("\n📝 Test 3: Name Memory (Recall)")
    print("-" * 40)
    response = engine.respond("내 이름 기억해?")
    print(f"User: 내 이름 기억해?")
    print(f"Elysia: {response}")
    assert "철수" in response, "Should recall saved name"
    print("✅ PASS: Name recalled from profile")
    
    # Test 4: Simple Math
    print("\n📝 Test 4: Simple Math")
    print("-" * 40)
    response = engine.respond("1+1은?")
    print(f"User: 1+1은?")
    print(f"Elysia: {response}")
    assert "2" in response, "Should calculate 1+1=2"
    print("✅ PASS: Math calculation works")
    
    # Test 5: Emotional Expression
    print("\n📝 Test 5: Emotional Expression")
    print("-" * 40)
    response = engine.respond("고마워!")
    print(f"User: 고마워!")
    print(f"Elysia: {response}")
    assert any(emoji in response for emoji in ["💚", "✨", "😊"]), "Should express gratitude with emoji"
    print("✅ PASS: Emotional expression works")
    
    # Test 6: Question Analysis
    print("\n📝 Test 6: Question Analysis")
    print("-" * 40)
    response = engine.respond("너는 누구야?")
    print(f"User: 너는 누구야?")
    print(f"Elysia: {response}")
    assert "Elysia" in response or "엘리시아" in response, "Should introduce herself"
    print("✅ PASS: Question understanding works")
    
    # Test 7: Memory Check
    print("\n📝 Test 7: Memory Persistence")
    print("-" * 40)
    exp_count = len(engine.memory.experience_loop)
    print(f"Experience Loop: {exp_count} entries")
    assert exp_count > 0, "Should have saved experiences"
    print("✅ PASS: Memories stored in Hippocampus")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("\n📊 Summary:")
    print(f"   ✅ Simple greetings: Working")
    print(f"   ✅ Name memory: Working")
    print(f"   ✅ Emotional expression: Working")
    print(f"   ✅ Question analysis: Working")
    print(f"   ✅ Math calculations: Working")
    print(f"   ✅ Memory persistence: Working")
    print(f"\n   💾 User Profile: {engine.user_profile}")
    print(f"   🧠 Hippocampus: {len(engine.memory.experience_loop)} experiences")
    print("=" * 60)


def test_conversation_flow():
    """Test natural conversation flow."""
    
    print("\n\n💬 Natural Conversation Test")
    print("=" * 60)
    
    engine = DialogueEngine()
    
    conversation = [
        "안녕?",
        "내 이름은 민수야",
        "1+1은?",
        "너는 뭐해?",
        "내 이름 기억해?",
        "고마워!"
    ]
    
    for user_input in conversation:
        response = engine.respond(user_input)
        print(f"\n👤 User: {user_input}")
        print(f"🤖 Elysia: {response}")
    
    print("\n" + "=" * 60)
    print("✅ Conversation flow complete!")
    print(f"💾 Profile: {engine.user_profile}")
    print(f"🧠 Memories: {len(engine.memory.experience_loop)} experiences")


if __name__ == "__main__":
    try:
        test_dialogue_improvements()
        test_conversation_flow()
        
        print("\n\n🎊 ALL IMPROVEMENTS WORKING PERFECTLY! 🎊")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
