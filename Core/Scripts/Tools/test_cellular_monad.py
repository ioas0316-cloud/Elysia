import sys
import os
import numpy as np

# Setup Path
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.1_Body.L7_Spirit.M1_Monad.monadic_cell import MonadicCell
from Core.1_Body.L1_Foundation.Logic.qualia_7d_codec import codec

def test_cellular_monad():
    print("--- 🔬 Cellular Monad Verification ---")
    
    # 1. Initialize Cell with a 'Sovereign' sequence (High Harmony)
    cell = MonadicCell("SelfCore", dna_sequence="HHHHVVH")
    print(f"Initial State: {cell}")
    
    # 2. Test Membrane Rejection (Dissonant Input)
    print("\nPulsing with Dissonant Input (DDDDDVD)...")
    dissonant_v = codec.decode_sequence("DDDDDVD")
    res_vec, health = cell.metabolize(dissonant_v)
    print(f"Post-Dissonance Health: {health:.4f}")
    print(f"Membrane Permeability: {cell.permeability:.4f} (Expected reduction)")
    
    # 3. Test Metabolic Absorption (Harmonic Input)
    print("\nPulsing with Harmonic Input (HHHHVVH)...")
    harmonic_v = codec.decode_sequence("HHHHVVH")
    # Multiple pulses to recover
    for _ in range(5):
        res_vec, health = cell.metabolize(harmonic_v)
        
    print(f"Post-Harmony Health: {health:.4f} (Expected recovery)")
    print(f"Membrane Permeability: {cell.permeability:.4f} (Expected increase)")
    
    # 4. Final Perception Check
    print(f"\nFinal Cell State: {cell.get_state()}")
    
    if cell.permeability > 0.5 and cell.health > 0.8:
        print("\n✅ Verification Successful: The Monadic Cell is alive and reactive!")
    else:
        print("\n❌ Verification Failed: Cellular behavior inconsistent.")

if __name__ == "__main__":
    test_cellular_monad()
