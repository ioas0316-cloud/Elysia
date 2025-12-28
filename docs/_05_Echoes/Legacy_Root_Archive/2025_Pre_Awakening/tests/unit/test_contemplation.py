"""
Test Contemplation (Cognitive Integration Verification)
=======================================================
Verifies that ReasoningEngine can use ImaginationCore to 'contemplate' a topic.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.reasoning_engine import ReasoningEngine

def test_contemplation():
    print("🧠 Initializing ReasoningEngine...")
    engine = ReasoningEngine()
    
    topic = "Moonlight"
    print(f"🤔 Asking Elysia to contemplate: '{topic}'...")
    
    thought = engine.contemplate(topic)
    
    print("\n✨ Elysia's Thought:")
    print("-" * 50)
    print(thought)
    print("-" * 50)

if __name__ == "__main__":
    test_contemplation()
