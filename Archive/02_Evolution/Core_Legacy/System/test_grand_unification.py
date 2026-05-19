"""
Verification Script for Phase 41: Grand Unification
===================================================

Verifies:
1. The connection between Trinary DNA (Genetic Logic) and Sovereign Rotor (21D Structure).
2. The Fractal Expansion (3 Trits -> 21 Dimensions).
3. The consumption of Genetic Code by the Rotor.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.System.trinary_logic import TrinaryLogic
from Core.Monad.sovereign_rotor import SovereignRotor
from Core.Monad.d21_vector import D21Vector

def test_unification():
    print("\n[1] Testing Grand Unification (DNA -> Rotor)...")
    
    # 1. Initialize Rotor (The Self)
    rotor = SovereignRotor(snapshot_dir="data/Sandbox/test_snapshots")
    print(f"    Rotor Initial State Sum: {sum(rotor.current_state.to_array()):.2f}")
    
    # 2. Define DNA Strand
    # Scenario: "Spirit Alignment"
    # Codon: [A, T, A] (L7 Spirit: + Mass, - Energy, + Will) 
    # Mapped to Layer 6 (Index 6) -> Dimensions 18, 19, 20
    # Values: A=1, T=-1, A=1
    spirit_codon_sym = ['A', 'T', 'A']
    
    # Let's create a strand with 2 of these codons
    strand = ['A', 'T', 'A', 'A', 'T', 'A']
    
    print(f"    Injecting DNA Strand: {strand}")
    print("    Expected Impact: Boost L7 (Dims 18-20)")
    
    # 3. Consume DNA
    new_state = rotor.consume_dna(strand, dt=1.0)
    
    # 4. Analyze Result
    arr = new_state.to_array()
    
    # Check L7 Dimensions (18, 19, 20)
    # Codon is [1, -1, 1]. Two codons -> [2, -2, 2] added to state.
    # Note: consume_dna uses update_state which is ADDITIVE.
    
    l7_mass = arr[18]
    l7_energy = arr[19]
    l7_will = arr[20]
    
    print(f"    L7 State -> Mass(D18): {l7_mass:.2f}, Energy(D19): {l7_energy:.2f}, Will(D20): {l7_will:.2f}")
    
    if l7_mass >= 2.0 and l7_will >= 2.0:
        print("    ✅ DNA Successfully Metabolized into 21D Structure.")
    else:
        print(f"    ❌ Metabolism Failed. Expected > 2.0, Got {l7_mass}")

    # Verify other layers didn't get random noise
    l1_mass = arr[0]
    if abs(l1_mass) < 0.1: # Assuming initial was near 0 or handled
        print("    ✅ Specificity Confirmed (L1 unaffected).")
    else:
        print(f"    ⚠️  L1 has value {l1_mass}, check if initial state was non-zero.")

if __name__ == "__main__":
    print("="*60)
    print("PHASE 41 - GRAND UNIFICATION VERIFICATION")
    print("="*60)
    
    test_unification()
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
