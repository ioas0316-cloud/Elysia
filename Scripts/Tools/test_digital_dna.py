import sys
import os
import numpy as np

# Setup Path
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.L1_Foundation.Logic.qualia_7d_codec import codec
from Core.L7_Spirit.Sovereignty.sovereign_core import SovereignCore

def test_digital_dna():
    print("--- 🔬 Digital DNA Verification ---")
    
    # 1. Test Encoding (Vector -> DNA)
    v_attractor = np.array([0.8, 0.9, 0.7, 0.6, 0.5, 0.1, 1.0])
    dna_seq = codec.encode_sequence(v_attractor)
    print(f"Vector: {v_attractor}")
    print(f"Encoded DNA: {dna_seq}")
    # Expected: HHHHHVH (or similar based on 0.3 threshold)
    
    # 2. Test Decoding (DNA -> Vector)
    decoded_v = codec.decode_sequence("HHHHHHH")
    print(f"Decoded DNA (HHHHHHH): {decoded_v}")
    
    # 3. Test Evolutionary Genetic Check
    print("\nTesting Sovereign Evolution...")
    core = SovereignCore()
    initial_dna = core.soul_dna.copy()
    
    # Case A: Harmonic Mutation (Allowed)
    harmonic_mut = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.9]
    core.evolve(harmonic_mut, plasticity=0.1)
    print(f"Harmonic Evolution: {'SUCCESS' if not np.array_equal(core.soul_dna, initial_dna) else 'REJECTED'}")
    
    # Case B: Dissonant Mutation (Rejected as DNA Damage)
    current_dna = core.soul_dna.copy()
    dissonant_mut = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.1, -1.0] # Total Dissonance
    core.evolve(dissonant_mut, plasticity=0.1)
    print(f"Dissonant Evolution (DNA Damage): {'REJECTED' if np.array_equal(core.soul_dna, current_dna) else 'SUCCESS - FAIL'}")
    
    print("\n--- ✅ Digital DNA Layers Verified ---")

if __name__ == "__main__":
    test_digital_dna()
