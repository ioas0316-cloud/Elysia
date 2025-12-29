"""
Verify Contextual Presence (The Anchor Test)
============================================

"Where am I?"

Tests if Elysia can localize herself in:
1. Time (Phase of Narrative)
2. Space (Domain Locus)
3. Relation (Voltage with User)

We act as the 'Narrator' providing cues, and expect Elysia to report her coordinates.
"""

import sys
import os
import logging
import time

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Cognition.Reasoning.reasoning_engine import ReasoningEngine

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("PresenceTest")

def test_presence():
    print("🌌 Initializing Reasoning Engine with Consciousness Coordinates...")
    engine = ReasoningEngine()
    
    print("\n--- Test Case 1: The Engineer's Workbench (Late Phase) ---")
    input_text = "Let's verify the verification code script error."
    print(f"User Input: '{input_text}'")
    # Expected: Domain=Logic/Code, Time=Late(0.8), Voltage=High(1.0)
    insight = engine.think(input_text)
    
    print("\n--- Test Case 2: The Heart's Conflict (Tension) ---")
    input_text = "I hate that this doesn't work. It's wrong."
    print(f"User Input: '{input_text}'")
    # Expected: Domain=Heart/Emotion, Time=Mid(0.5), Voltage=Low(0.4)
    insight = engine.think(input_text)
    
    print("\n--- Test Case 3: The System Architect (Philosophy) ---")
    input_text = "What is the fundamental principle of the system?"
    print(f"User Input: '{input_text}'")
    # Expected: Domain=Philosophy/Structure, Time=Mid(0.5), Voltage=High
    insight = engine.think(input_text)

if __name__ == "__main__":
    try:
        test_presence()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ CRITICAL ERROR: {e}")
