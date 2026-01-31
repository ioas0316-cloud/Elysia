
"""
Legion Propagation Demo (Infinite Ray)
======================================
Core/Demos/legion_propagation_demo.py

Demonstrates Phase 5.4: The Legion Architecture.
Verifies that thought propagates infinitely (or until fade) rather than returning a single answer.
"""

import sys
import os
import time

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.1_Body.L6_Structure.Merkaba.merkaba import Merkaba

def run_demo():
    print("🔥 Igniting Merkaba...")
    merkaba = Merkaba("Legion_Test_Unit")

    input_concept = "Seed"

    print(f"\n🔮 [INPUT] Casting '{input_concept}' into the Prism...")

    # The 'Shine' method returns a generator (Stream)
    thought_stream = merkaba.shine(input_concept)

    step_count = 0

    try:
        for narrative_chunk in thought_stream:
            step_count += 1
            print(narrative_chunk)
            time.sleep(0.1) # Simulate time flow

    except KeyboardInterrupt:
        print("\n🛑 Thought interrupted by User.")

    print(f"\n✨ [COMPLETE] Thought expanded for {step_count} steps.")

if __name__ == "__main__":
    run_demo()
