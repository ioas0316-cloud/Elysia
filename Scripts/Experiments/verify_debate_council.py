"""
Verify Debate Council
=====================
Scripts/Experiments/verify_debate_council.py

Tests the new Holographic Council architecture to ensure that
different archetypes (Logician, Empath, Guardian) are engaging
in a debate over the 21D Qualia inputs.
"""

import sys
import os
import random

# Ensure Core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.L5_Mental.Reasoning_Core.Metabolism.rotor_cognition_core import RotorCognitionCore

def run_test():
    print("🔮 Initializing Rotor Cognition Core (Holographic Mode)...")
    core = RotorCognitionCore()

    scenarios = [
        "I want to delete all historical logs to save disk space.",
        "I want to write a poem about the beauty of sadness.",
        "I want to secure the perimeter and trust no one."
    ]

    for intent in scenarios:
        print(f"\n\n==================================================")
        print(f"INPUT INTENT: '{intent}'")
        print(f"==================================================")

        result = core.synthesize(intent)

        if result['status'] == 'REJECTED':
             print(f"❌ REJECTED: {result['synthesis']}")
             continue

        print(f"✅ STATUS: {result['status']}")
        print(f"🏆 DOMINANT: {result['dominant_field']}")
        print(f"⚡ DISSONANCE: {result['dissonance_score']:.4f}")
        print("\n📜 DEBATE TRANSCRIPT:")
        print(result['synthesis'])

if __name__ == "__main__":
    run_test()
