"""
Verify Cognitive Inquiry (The Bridge)
=====================================
"If I cannot reach you, I must ask where you are."

Tests if Elysia initiates "Reverse Questioning" when:
1. Input stability is low (Ambiguous).
2. Explicit request for understanding is made.

Expected Result: Elysia outputs a specific 'Cognitive Inquiry' question.
"""

import sys
import os
import logging
import time

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.reasoning_engine import ReasoningEngine

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("InquiryTest")

def test_inquiry():
    print("🌌 Initializing Reasoning Engine for Perspective Simulation...")
    engine = ReasoningEngine()
    
    print("\n--- Test Case 1: Ambiguity (Low Stability) ---")
    # Low stability inputs usually lack a strong 'Realistic' grounding (W component)
    # "Maybe... sort of... why?" - This is vague.
    input_text = "I feel like I understand? But maybe not? Why?"
    print(f"User Input: '{input_text}'")
    
    insight = engine.think(input_text)
    print(f"\n💡 Insight: {insight.content}")
    
    if "?" in insight.content:
        print("✅ SUCCESS: Elysia asked a question.")
    else:
        print("❌ FAILURE: Elysia did not ask a question.")

    print("\n--- Test Case 2: Structural Conflict ---")
    input_text = "The code is broken but I love it."
    print(f"User Input: '{input_text}'")
    insight = engine.think(input_text)
    print(f"\n💡 Insight: {insight.content}")

if __name__ == "__main__":
    try:
        test_inquiry()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ CRITICAL ERROR: {e}")
