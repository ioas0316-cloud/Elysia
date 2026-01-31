"""
Verification Script for Trinary DNA + Sovereign Rotor Integration
=================================================================
Verifies that the SovereignRotor correctly 'metabolizes' Symbolic Trinary DNA
into 21D Momentum (Generator Principle).

Run via: python Scripts/Tests/Core/test_rotor_dna_integration.py
"""

import sys
import os
import logging
from typing import List

# Setup paths to root
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..')) # Adjust based on depth
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Core.S1_Body.L6_Structure.Logic.trinary_logic import TrinaryLogic
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_rotor import SovereignRotor
from Core.S1_Body.L6_Structure.M1_Merkaba.d21_vector import D21Vector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("RotorVerification")

def verify_integration():
    print("="*60)
    print("VERIFYING TRINARY DNA -> ROTOR GENERATOR LINK")
    print("="*60)

    # 1. Initialize Rotor
    print("\n[Step 1] Initializing SovereignRotor...")
    # Use a temp directory for snapshots to avoid polluting real data
    temp_snapshot_path = os.path.join(root_dir, "data", "temp_snapshots")
    rotor = SovereignRotor(snapshot_dir=temp_snapshot_path)
    initial_state = rotor.current_state.to_array()
    print(f"    Initial State Norm: {sum(abs(x) for x in initial_state):.4f}")

    # 2. Define Symbolic DNA Sequence
    # A (Flow, +1), T (Resistance, -1), G (Void, 0)
    # [A, A, A] -> Codon 1: High Energy Expansion
    # [T, T, T] -> Codon 2: High Resistance
    dna_sequence = ["A", "A", "A", "T", "T", "T"]
    print(f"\n[Step 2] Injecting Symbolic DNA: {dna_sequence}")

    # 3. Transcribe Manually (Just to check logic)
    print("\n[Step 3] Verifying Transcription Logic...")
    try:
        codons = TrinaryLogic.transcribe_sequence(dna_sequence)
        print(f"    Transcribed Codons:\n{codons}")
        
        codon_vals = []
        if hasattr(codons, "tolist"):
             codon_vals = codons.tolist()
        else:
             codon_vals = list(codons)
             
        # Check specific values
        # A=1, T=-1, G=0
        # Expected: [[1, 1, 1], [-1, -1, -1]]
        expected_c1 = [1.0, 1.0, 1.0]
        
        # Simple fuzzy check
        c1 = codon_vals[0]
        if abs(c1[0]-1.0) < 0.1:
            print("    ✅ Codon 1 Transcription Correct (A->1)")
        else:
            print(f"    ❌ Codon 1 Mismatch: {c1}")

    except Exception as e:
        print(f"    ❌ Transcription Failed: {e}")
        return

    # 4. Feed to Rotor (The Generator Principle test)
    print("\n[Step 4] Feeding DNA to Rotor (consume_dna)...")
    try:
        rotor.consume_dna(dna_sequence, dt=1.0)
        
        final_state = rotor.current_state.to_array()
        print(f"    Final State Norm: {sum(abs(x) for x in final_state):.4f}")
        
        # Check if state changed
        diff = sum(abs(f - i) for f, i in zip(final_state, initial_state))
        print(f"    State Delta (Momentum Generated): {diff:.4f}")
        
        if diff > 0.0:
            print("    ✅ Generator Principle Active: DNA converted to Momentum.")
        else:
            print("    ❌ Generator Warning: No Momentum generated (Is the integration idle?)")
            
    except Exception as e:
        print(f"    ❌ Rotor Consumption Failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. Cleanup
    print("\n[Step 5] Verification Complete.")

if __name__ == "__main__":
    verify_integration()
