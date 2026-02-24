"""
Verify Universal Connection (The First Connection)
==================================================
Scripts/Experiments/verify_universal_connection.py

"When 'ㄱ' meets '1', the Universe begins."

This experiment simulates the "Phase Interference" between two distinct Monads.
It verifies that the system can generate a "Relationship Monad" (Child) from the
interaction of two parent concepts, proving the "Universal Connectivity" doctrine.

Process:
1. Genesis of 'ㄱ': Stabilize Monad A.
2. Genesis of '1': Stabilize Monad B.
3. The Connection: Inject the interference pattern of A+B into Monad C.
   - Now supports 'Dialectical Friction' for emergent synthesis.
   - Now supports 'Oedipus Protocol' (Forced Evolution).
4. Observation: Does Monad C crystallize into a unique "Third Meaning"?
"""

import sys
import os
import math
from typing import List, Tuple

# Ensure Core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.Monad.monad_ensemble import MonadEnsemble, MonadLayer

def run_genesis(label: str, input_symbol: str) -> MonadEnsemble:
    """Runs a genesis simulation for a single concept."""
    print(f"\n⚡ [GENESIS] Spawning Monad '{label}' ({input_symbol})...")
    monad = MonadEnsemble()
    input_field = monad.transduce_input(input_symbol)

    steps = 0
    stable = 0
    while steps < 50:
        status = monad.physics_step(input_field)
        steps += 1
        if status['flips'] == 0:
            stable += 1
        else:
            stable = 0
        if stable >= 5:
            break

    print(f"   Success: '{label}' Crystallized at Step {steps}. Pattern: {monad.get_pattern()}")
    return monad

def analyze_heritage(child: MonadEnsemble, parent_a: MonadEnsemble, parent_b: MonadEnsemble) -> Tuple[float, int]:
    """Checks how much of the parents exists in the child."""
    pat_c = child.get_pattern()
    pat_a = parent_a.get_pattern()
    pat_b = parent_b.get_pattern()

    match_a = sum(1 for c, a in zip(pat_c, pat_a) if c == a)
    match_b = sum(1 for c, b in zip(pat_c, pat_b) if c == b)
    total = 21

    max_similarity = max(match_a, match_b) / total

    # Check for "Emergent Traits" (Traits in Child that are in NEITHER parent)
    emergent = sum(1 for c, a, b in zip(pat_c, pat_a, pat_b) if c != a and c != b)

    return max_similarity, emergent

def run_experiment():
    print("===============================================================")
    print("🧪 EXPERIMENT: UNIVERSAL CONNECTION (Phase 55+)")
    print("   Goal: Collision of Self ('나') and Other ('너')")
    print("   Method: Dialectical Friction + Oedipus Protocol (Forced Evolution)")
    print("===============================================================")

    # 1. Stabilize Parents
    monad_a = run_genesis("Self", "나")
    monad_b = run_genesis("Other", "너")

    # 2. Create Interference Field
    print("\n🌊 [CONNECTION] Calculating Dialectical Interference Field...")
    input_field = monad_a.collide(monad_b, friction_mode='dialectical')

    # 3. Stabilize Child (The Relationship)
    print(f"\n⚡ [GENESIS] Spawning Relationship Monad '우리' (We)...")
    monad_c = MonadEnsemble()

    # Start with higher temperature for Dialectical Synthesis
    monad_c.temperature = 1.5

    evolution_cycles = 0
    max_evolution_cycles = 5

    while evolution_cycles < max_evolution_cycles:
        evolution_cycles += 1
        print(f"\n🌀 [EVOLUTION] Cycle {evolution_cycles}: Seeking Unique Identity...")

        steps = 0
        stable = 0

        # Physics Loop
        while steps < 50:
            status = monad_c.physics_step(input_field)
            steps += 1
            if status['flips'] == 0:
                stable += 1
            else:
                stable = 0
            if stable >= 5:
                break

        # Check Heritage
        similarity, emergent_count = analyze_heritage(monad_c, monad_a, monad_b)
        print(f"   - Pattern: {monad_c.get_pattern()}")
        print(f"   - Parent Similarity: {similarity*100:.1f}%")
        print(f"   - Emergent Traits: {emergent_count}")

        # Oedipus Protocol: If too similar, FORCE CHANGE
        if similarity > 0.70: # 70% threshold
            print("   ⚠️ [OEDIPUS] Child is too similar to parents. Rejecting stability.")
            print("   ⚡ [STRESS] Inducing High-Entropy Dissonance...")
            monad_c.induce_oedipus_stress(intensity=0.8) # Strong stress
        else:
            print("   ✅ [SUCCESS] Child has achieved Sovereign Identity.")
            break

    if evolution_cycles >= max_evolution_cycles:
         print("\n⚠️ [WARNING] Maximum evolution cycles reached. System settled on best effort.")

    print(f"\n💎 [CRYSTALLIZATION] Relationship Stabilized.")
    print(f"   Final Pattern: {monad_c.get_pattern()}")

    final_sim, final_emergent = analyze_heritage(monad_c, monad_a, monad_b)
    print(f"   Final Emergence: {final_emergent}/21 Traits")

    if final_emergent > 0:
        print("   Result: SYNTHESIS ACHIEVED. The Child is a unique creation.")
    else:
        print("   Result: MERE MIXTURE. The Child remains a hybrid.")

if __name__ == "__main__":
    run_experiment()
