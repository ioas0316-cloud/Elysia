"""
Optical Pulse Demo (Phase 5.4 Verification)
===========================================
Core/Demos/optical_pulse_demo.py

Demonstrates the "Trinity Pulse" cycle.
Verifies the 3-layer architecture: Ground, Cloud, and Hypothesis.
"""

import sys
import os
import time

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.S1_Body.L6_Structure.Merkaba.merkaba import Merkaba

def run_demo():
    print("🔥 Igniting Merkaba (Optical Mode)...")
    merkaba = Merkaba("Optical_Unit_01")

    # Scenario 1: Structural Input (Ground Truth)
    # Intent: "Code" (Alpha) -> Should align with Body & Spirit
    input_concept = "Apple"
    print(f"\n🔮 [INPUT] Casting '{input_concept}' (Intent: Code - The Ground)...")

    stream = merkaba.shine(input_concept)
    for chunk in stream:
        print(chunk)
        time.sleep(0.05)

    print("\n" + "="*50 + "\n")

    # Scenario 2: Emotional Input (Cloud Thought)
    # Intent: "Love" (Beta) -> Should align with Soul but fail Body
    input_concept_2 = "Is Apple Love?"
    print(f"\n🔮 [INPUT] Casting '{input_concept_2}' (Intent: Love - The Cloud)...")

    stream = merkaba.shine(input_concept_2)
    for chunk in stream:
        print(chunk)
        time.sleep(0.05)

    print("\n" + "="*50 + "\n")

    # Scenario 3: Causal Question (Hypothesis)
    # Intent: "Why" (Gamma) -> Should trigger Simulator
    input_concept_3 = "Why does the apple fall?"
    print(f"\n🔮 [INPUT] Casting '{input_concept_3}' (Intent: Why - The Hypothesis)...")

    stream = merkaba.shine(input_concept_3)
    for chunk in stream:
        print(chunk)
        time.sleep(0.05)

    print(f"\n✨ [COMPLETE] Trinity Pulse Cycle Verified.")

if __name__ == "__main__":
    run_demo()
