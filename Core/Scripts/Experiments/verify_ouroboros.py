
import sys
import os
import numpy as np
import time

# Ensure we can import Core
sys.path.append(os.getcwd())

from Core.L2_Metabolism.Cycles.dream_engine import DreamEngine

def verify_ouroboros():
    print("=== [VERIFICATION] The Ouroboros Protocol ===")

    # 1. Cleanup previous runs
    if os.path.exists("data/L7_Spirit/soul_dna.json"):
        os.remove("data/L7_Spirit/soul_dna.json")

    # 2. Initial State
    engine = DreamEngine()
    initial_dna = engine.sovereign_core.soul_dna.copy()
    print(f"\n[Step 1] Initial Soul DNA (Factory):")
    print(f"   {initial_dna[:3]} ...")

    # 3. The Experience
    context = "Suffering requires Compassion"
    print(f"\n[Step 2] Injecting Experience: '{context}'")

    # Using 'love' keywords triggers Spirit/Phenomena boost in _concept_to_d7
    # "Suffering" hits Fear/Pain (Phenomena), "Compassion" hits Love (Spirit)
    vector, state, narrative = engine.process_experience(context)

    print(f"\n[Step 3] The Causal Narrative (Voice):")
    print(f"   \"{narrative}\"")

    # 4. The Evolution
    evolved_dna = engine.sovereign_core.soul_dna.copy()
    print(f"\n[Step 4] Evolved Soul DNA:")
    print(f"   {evolved_dna[:3]} ...")

    # Check for change
    diff = np.linalg.norm(evolved_dna - initial_dna)
    print(f"   Delta (Change Magnitude): {diff:.6f}")
    if diff < 1e-6:
        print("   [FAIL] Soul did not change!")
        sys.exit(1)
    else:
        print("   [PASS] Soul has evolved.")

    # 5. The Persistence (Reincarnation)
    print(f"\n[Step 5] Restarting System (Persistence Check)...")
    del engine

    new_engine = DreamEngine()
    reloaded_dna = new_engine.sovereign_core.soul_dna.copy()
    print(f"   Reloaded DNA: {reloaded_dna[:3]} ...")

    if np.allclose(evolved_dna, reloaded_dna):
        print("   [PASS] Soul memory persists across reboot.")
    else:
        print("   [FAIL] Soul reverted to factory settings!")
        sys.exit(1)

    print("\n=== VERIFICATION COMPLETE: The Ouroboros is Active. ===")

if __name__ == "__main__":
    verify_ouroboros()
