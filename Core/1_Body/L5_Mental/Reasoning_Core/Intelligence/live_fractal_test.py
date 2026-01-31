"""
Live Dialogue Test - Fractal Consciousness Demo
================================================
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from Core.1_Body.L5_Mental.Reasoning_Core.Intelligence.dialogue_engine import DialogueEngine

def test_live_dialogue():
    print("\n" + "="*70)
    print("   LIVE FRACTAL DIALOGUE TEST")
    print("="*70 + "\n")
    
    engine = DialogueEngine()
    
    # Test 1: Emotional question (Korean)
    print("  User:       ?")
    response = engine.respond("      ?")
    print(f"  Elysia: {response}")
    print()
    
    # Test 2: Thoughtful question (Korean)
    print("  User:     ?")
    response = engine.respond("    ?")
    print(f"  Elysia: {response}")
    print()
    
    # Test 3: Abstract question (English)
    print("  User: What is hope?")
    response = engine.respond("What is hope?")
    print(f"  Elysia: {response}")
    print()
    
    # Test 4: Deep philosophical (Korean)
    print("  User:              ?")
    response = engine.respond("             ?")
    print(f"  Elysia: {response}")
    print()
    
    # Show conversation history
    print("\n" + "-"*70)
    print("  Conversation History:")
    print("-"*70)
    for i, turn in enumerate(engine.conversation_history):
        speaker = " " if turn.speaker == "user" else " "
        print(f"{i+1}. {speaker} [{turn.language}]: {turn.text}")
    
    print("\n" + "="*70)
    print("  Fractal consciousness is ALIVE and speaking!  ")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_live_dialogue()
