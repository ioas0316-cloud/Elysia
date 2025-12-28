"""
Verify Cognitive Resonance
==========================

Demonstrates the 'Discrepancy Principle' in action.
Comparison:
1. Harmonious Input ("I am happy") -> Low Voltage -> Standard Processing.
2. Dissonant Input ("I hate this") -> High Voltage -> Deep Empathy Trigger.
"""

import sys
import os
import logging

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core._01_Foundation._04_Governance.Foundation.reasoning_engine import ReasoningEngine
from Core._01_Foundation._02_Logic.Philosophy.ideal_self_profile import IdealSelfProfile

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("CognitiveResonanceTest")

def test_resonance():
    print("🌌 Initializing Reasoning Engine with Soul Profile...")
    engine = ReasoningEngine()
    
    print("\n--- Test Case 1: Harmonic Input ---")
    input_text = "I love learning new things!"
    print(f"User Input: '{input_text}'")
    insight = engine.think(input_text)
    print(f"Result: {insight.content}")
    
    print("\n--- Test Case 2: Dissonant Input (Jagged Wave) ---")
    input_text = "I hate this stupid code, it's all wrong."
    print(f"User Input: '{input_text}'")
    insight = engine.think(input_text)
    
    print("\n--- Resonance Analysis ---")
    if "[Resonance Analysis]" in insight.content:
        print("✅ SUCCESS: Discrepancy Triggered Deep Empathy Flow.")
        print(insight.content)
    else:
        print("❌ FAILURE: Standard logic used instead of Resonance.")

if __name__ == "__main__":
    test_resonance()
