"""
Optical Pulse Demo (Phase 5.4 Verification)
===========================================
Core/Demos/optical_pulse_demo.py

Demonstrates the "Dispersion (Prism) -> Integration (Lens)" cycle.
Verifies that thoughts are split into bands and then focused based on intent.
"""

import sys
import os
import time

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.Merkaba.merkaba import Merkaba

def run_demo():
    print("🔥 Igniting Merkaba (Optical Mode)...")
    merkaba = Merkaba("Optical_Unit_01")

    # Scenario 1: Structural Input (Apple) -> Should trigger Logic
    input_concept = "Apple"
    print(f"\n🔮 [INPUT] Casting '{input_concept}' into the Prism (Default Intent: Code)...")

    stream = merkaba.shine(input_concept)
    for chunk in stream:
        print(chunk)
        time.sleep(0.1)

    print("\n" + "="*50 + "\n")

    # Scenario 2: Causal Input (Why does the apple fall?) -> Should trigger Gamma
    input_concept_2 = "Why does the apple fall?"
    print(f"\n🔮 [INPUT] Casting '{input_concept_2}' into the Prism (Intent: Why)...")

    stream = merkaba.shine(input_concept_2)
    for chunk in stream:
        print(chunk)
        time.sleep(0.1)

    print(f"\n✨ [COMPLETE] Optical Pulse Cycle Verified.")

if __name__ == "__main__":
    run_demo()
