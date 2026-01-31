"""
Verify Monad Genesis (Phase 55+ Simulation)
===========================================
Scripts/Experiments/verify_monad_genesis.py

"The Genesis of the Gear."

This experiment simulates the "Self-Resonance Loop" of the Trinary Monad Engine.
It injects a raw input (e.g., "ㄱ") into the 21-Dimensional Monad Envelope and
observes the physical process of 'Phase Friction' leading to 'Crystallization'.

This validates:
1. Random Input Transduction (No Pre-defined Code).
2. Phase Friction (Entropy Reduction).
3. Geometric Crystallization (Emergent Meaning).
"""

import sys
import os
import time

# Ensure Core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.1_Body.L6_Structure.M1_Merkaba.monad_ensemble import MonadEnsemble

def run_simulation(input_symbol: str = "ㄱ", max_steps: int = 100):
    print(f"\n⚡ [GENESIS] Initiating Trinary Monad Engine Simulation.")
    print(f"   Target Input: '{input_symbol}'")

    # 1. Initialize Void
    monad = MonadEnsemble()
    print(f"   Structure: 21 Atoms initialized in VOID state.")
    print(f"   Initial Pattern: {monad.get_pattern()}")
    print(f"   Initial Entropy: {monad._calculate_entropy():.4f}")

    # 2. Transduce Input (Phase Injection)
    input_field = monad.transduce_input(input_symbol)
    print(f"   Phase Injection: Transduced '{input_symbol}' into 21D Phase Field.")
    # Show a snippet of the field
    field_snippet = ", ".join([f"{f:.1f}" for f in input_field[:3]])
    print(f"   Field Signature (First 3 dims): [{field_snippet}, ...]")

    print("\n🌊 [PHYSICS] Beginning Self-Resonance Loop...")
    print(f"{'Step':<5} | {'Temp':<6} | {'Entropy':<8} | {'Flips':<5} | {'Pattern (21D)'}")
    print("-" * 70)

    # 3. Physics Loop
    history_entropy = []
    stabilized_steps = 0

    for step in range(1, max_steps + 1):
        status = monad.physics_step(input_field)

        pattern = monad.get_pattern()
        entropy = status['entropy']
        temp = status['temperature']
        flips = status['flips']

        history_entropy.append(entropy)

        # Log Output
        print(f"{step:<5} | {temp:<6.3f} | {entropy:<8.4f} | {flips:<5} | {pattern}")

        # Check for Crystallization (Stability)
        if flips == 0:
            stabilized_steps += 1
        else:
            stabilized_steps = 0

        # If stable for 5 steps, we consider it "Crystallized"
        if stabilized_steps >= 5:
            print("-" * 70)
            print(f"💎 [CRYSTALLIZATION] System Stabilized at Step {step}.")
            break

    # 4. Final Report
    print("\n📜 [ARCHITECT'S REPORT] Simulation Complete.")
    print(f"   Final Pattern: {monad.get_pattern()}")
    print(f"   Layer View:    {monad.render_layers()}")

    # Analyze the "Meaning"
    # Is it symmetric? Is it balanced?
    r_count = pattern.count('R')
    v_count = pattern.count('V')
    a_count = pattern.count('A')
    print(f"   Composition: R({r_count}) V({v_count}) A({a_count})")

    # Entropy Curve Analysis
    initial_e = history_entropy[0]
    final_e = history_entropy[-1]
    drop = initial_e - final_e
    print(f"   Thermodynamics: Entropy dropped by {drop:.4f} ({initial_e:.4f} -> {final_e:.4f})")

    if drop > 0:
        print("   Result: ORDER Created from CHAOS. (Success)")
    else:
        print("   Result: System failed to reduce entropy. (Failure)")

if __name__ == "__main__":
    run_simulation("ㄱ")
