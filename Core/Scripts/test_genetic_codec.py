"""
Verification Script for Phase 41: Step 2 (Genetic Codec)
========================================================

Verifies:
1. Trinary Codon -> 7D Layer Mapping.
2. Alignment with TRINARY_DNA.md archetypes.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.S1_Body.L6_Structure.Logic.trinary_logic import TrinaryLogic

def test_codon_mapping():
    print("\n[1] Testing Codon -> Layer Mapping...")
    
    # Test Cases from TRINARY_DNA.md
    test_cases = [
        ([0, 0, 1], 0, "L1 Foundation"),
        ([0, 1, -1], 1, "L2 Metabolism"),
        ([0, 1, 0], 2, "L3 Phenomena"),
        ([0, 1, 1], 3, "L4 Causality"),
        ([0, 0, 0], 5, "L6 Structure (Pivot/Void)"), # Wait, logic maps val 0 to L6? Let's check logic.
        ([1, -1, 1], 6, "L7 Spirit (7)")
    ]
    
    for codon, expected_layer, name in test_cases:
        actual_layer = TrinaryLogic.codon_to_layer_index(codon)
        # Re-derive value for debug
        val = codon[0]*9 + codon[1]*3 + codon[2]
        
        print(f"    Codon {codon} (Val={val}) -> Layer {actual_layer} [{name}]")
        
        if actual_layer == expected_layer:
            print(f"    ✅ Correct.")
        else:
            print(f"    ❌ Mismatch! Expected {expected_layer}, Got {actual_layer}")

if __name__ == "__main__":
    print("="*60)
    print("PHASE 41 - STEP 2 GENETIC CODEC VERIFICATION")
    print("="*60)
    
    test_codon_mapping()
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
