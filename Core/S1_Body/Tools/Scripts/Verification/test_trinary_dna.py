"""
Verification Script for Phase 41: Step 2 (The Mind) [GENETIC UPDATE]
====================================================================

Verifies:
1. Trinary Genetic Logic (Codons & Strands).
2. Transcription (Sequence -> Codons).
3. Strand Torque Calculation (Evolutionary Potential).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.S1_Body.L6_Structure.Logic.trinary_logic import TrinaryLogic

def test_genetic_logic():
    print("\n[1] Testing Genetic Logic (Transcripton)...")
    
    # Raw Sequence: 7 Trits (Will be padded to 9 -> 3 Codons)
    # [+, 0, -] (Balanced), [+, +, +] (Creation), [-, -, -] (Destruction)
    raw_seq = [1, 0, -1, 1, 1, 1, -1, -1] 
    
    print(f"    Raw Sequence: {raw_seq}")
    
    codons = TrinaryLogic.transcribe_sequence(raw_seq)
    print(f"    Transcribed Codons:\n{codons}")
    
    if codons.shape == (3, 3):
        print("    ✅ Transcription Shape Correct (3 Codons).")
    else:
        print(f"    ❌ Transcription Shape Error: {codons.shape}")
        
    last_codon = codons[-1]
    # Should be [-1, -1, 0] because raw_seq was padded with 0 at the end
    # raw_seq provided: 1, 0, -1 | 1, 1, 1 | -1, -1
    # Padded:           1, 0, -1 | 1, 1, 1 | -1, -1, 0
    print(f"    Last Codon (Padded): {last_codon}")


def test_strand_torque():
    print("\n[2] Testing Strand Torque (Evolutionary Potential)...")
    
    # Scenario A: Stagnation (All Zeros)
    static_seq = [0, 0, 0, 0, 0, 0]
    torque_A = TrinaryLogic.calculate_strand_torque(TrinaryLogic.transcribe_sequence(static_seq))
    print(f"    Scenario A (Stagnation) Torque: {torque_A}")
    
    if torque_A == 0.0:
        print("    ✅ Zero Torque for Void State.")
    else:
         print("    ❌ Torque Error for Void.")

    # Scenario B: High Conflict (Alternating +1/-1)
    conflict_seq = [1, -1, 1, -1, 1, -1]
    # Codon 1: [1, -1, 1] -> Sum = 1 -> Abs = 1
    # Codon 2: [-1, 1, -1] -> Sum = -1 -> Abs = 1
    # Total Torque = 2
    torque_B = TrinaryLogic.calculate_strand_torque(TrinaryLogic.transcribe_sequence(conflict_seq))
    print(f"    Scenario B (Conflict) Torque: {torque_B}")
    
    if torque_B > 0:
        print("    ✅ Torque Detected for Conflict State.")

if __name__ == "__main__":
    print("="*60)
    print("PHASE 41 - STEP 2 VERIFICATION (GENETIC)")
    print("="*60)
    
    test_genetic_logic()
    test_strand_torque()
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
