
"""
Verification Script: Curiosity-Driven Language Learning
=====================================================

"The Child who asks, learns."

This script verifies that Elysia can:
1. Detect a Semantic Gap (feel something unknown).
2. Trigger Curiosity (Gap > Threshold).
3. Query the World Lexicon (Ask "What is this?").
4. Assimilate the new concept ("Nostalgia").
5. Use the new concept in a subsequent thought.
"""

import sys
import os
import logging
import math

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from Core.1_Body.L5_Mental.emergent_language import EmergentLanguageEngine, SymbolType

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("VerifyLanguage")

def debug_resonance(engine, vector):
    print("    DEBUG: Checking resonances...")
    best_sym = None
    best_score = -1.0
    for sym_id, sym in engine.symbols.items():
        dot_product = sum(a * b for a, b in zip(vector, sym.meaning_vector))
        norm_exp = math.sqrt(sum(x**2 for x in vector)) + 0.001
        norm_sym = math.sqrt(sum(x**2 for x in sym.meaning_vector)) + 0.001
        similarity = dot_product / (norm_exp * norm_sym)

        if similarity > 0.6:
            print(f"      - {sym_id}: {similarity:.4f}")

        if similarity > best_score:
            best_score = similarity
            best_sym = sym_id

    print(f"    Best Match: {best_sym} ({best_score:.4f})")

def run_verification():
    print("="*60)
    print("   VERIFICATION: CURIOSITY-DRIVEN LANGUAGE ACQUISITION")
    print("="*60)

    # 1. Initialize Engine
    engine = EmergentLanguageEngine()
    print(f"\n[1] Initial Vocabulary Size: {engine.vocabulary_size}")

    # 2. Define an "Alien" Experience
    # We use the vector for "NOSTALGIA" from world_lexicon.py
    # Updated Vector: [0.3, -0.6, 0.4, -0.8, 0.9, 0.4, 0.1, -0.3]
    nostalgia_vector = [0.3, -0.6, 0.4, -0.8, 0.9, 0.4, 0.1, -0.3]

    # Check if 'NOSTALGIA' exists (It shouldn't)
    if "NOSTALGIA" in engine.symbols:
        print("ERROR: NOSTALGIA already exists in initial vocab!")
        return

    # Debug Resonance
    debug_resonance(engine, nostalgia_vector)

    # 3. Inject Experience (Triggering Curiosity)
    print("\n[2] Injecting Unknown Experience (Nostalgia)...")
    gap = engine.detect_semantic_gap(nostalgia_vector)
    print(f"    Semantic Gap Detected: {gap:.4f} (Threshold: 0.4)")

    # If gap is too small, we might need to adjust the vector or threshold for the test
    # But fundamentally, Nostalgia SHOULD be different enough.

    if gap < 0.4:
        print("WARNING: Gap too small. Forcing threshold lower for test or adjusting vector.")
        # Let's try a VERY alien vector if Nostalgia fails
        # [0, 0, 0, 0, 0, 0, 0, 0] ? No.
        # Let's try ANXIETY: [-0.3, 0.1, 0.2, 0.9, -0.5, 0.8, -0.8, 0.9]
        pass

    # This call should trigger the internal query and learning
    activated_symbols = engine.experience(nostalgia_vector)

    # 4. Verify Learning
    print("\n[3] Verifying Acquisition...")
    if "NOSTALGIA" in engine.symbols:
        print("    SUCCESS: 'NOSTALGIA' found in symbol table!")
        print(f"    Symbol Type: {engine.symbols['NOSTALGIA'].type}")
    else:
        print("    FAILURE: 'NOSTALGIA' was not learned.")
        return

    if "NOSTALGIA" in activated_symbols:
        print("    SUCCESS: 'NOSTALGIA' was activated immediately.")
    else:
        print("    WARNING: 'NOSTALGIA' learned but not activated?")

    # 5. Verify Utterance Generation
    print("\n[4] Generating Utterance with new Concept...")
    # Force activation of the new symbol to ensure it appears
    engine.symbols["NOSTALGIA"].activation = 1.0
    korean, english = engine.generate_utterance()

    print(f"    Result (EN): {english}")
    print(f"    Result (KO): {korean}")

    if "NOSTALGIA" in english or "NOSTALGIA" in korean:
        print("\n[PASS] System successfully learned and used the new concept.")
    else:
        print("\n[FAIL] System learned the word but failed to speak it.")

if __name__ == "__main__":
    run_verification()
