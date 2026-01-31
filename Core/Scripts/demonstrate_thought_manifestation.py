
import sys
import os
import math
import random

# Add root directory to sys.path
sys.path.append('c:/Elysia')

try:
    from Core.1_Body.L6_Structure.M1_Merkaba.d21_vector import D21Vector
    from Core.1_Body.L6_Structure.Logic.trinary_logic import TrinaryLogic
    from Core.1_Body.L1_Foundation.Foundation.hangul_physics import HangulPhysicsEngine, Tensor3D
except ImportError:
    print("Dependencies missing. Using mock logic.")
    class D21Vector:
        def __init__(self, data=None): self.data = data or [0.0]*21
        def to_array(self): return self.data
    class TrinaryLogic:
        @staticmethod
        def symbol_to_trit(s): return {'A':1, 'G':0, 'T':-1}.get(s, 0)

def demonstrate_manifestation():
    print("="*60)
    print("🧬 [DEMO] TRINARY DNA TO LINGUISTIC MANIFESTATION")
    print("="*60)

    # 1. Define distinct DNA sequences (Thoughts)
    # A sequence representing 'High Energy/Resonance'
    dna_love = "AAAAA" 
    # A sequence representing 'Resistance/Obstruction'
    dna_wrath = "TTTTT"

    eng = HangulPhysicsEngine()

    for name, dna in [("LOVE (Resonance)", dna_love), ("WRATH (Resistance)", dna_wrath)]:
        print(f"\n--- Stimulus: {name} (DNA: {dna}) ---")
        
        # 2. Convert DNA to Trits
        trits = [TrinaryLogic.symbol_to_trit(s) for s in dna]
        
        # 3. Simulate 21D Vector Impact
        # (Simplified: mapped to a 3D Tensor for Hangul Physics)
        mag = sum(trits)
        roughness = 0.1 if 'A' in dna else 0.9
        tension = 0.2 if 'A' in dna else 0.8
        
        tensor = Tensor3D(x=float(trits[0])*5, y=float(trits[1])*5, z=float(mag))
        
        # 4. Generate Linguistic Units (Babbling)
        onset = eng.find_closest_jamo(roughness, tension, 'consonant')
        nucleus = eng.find_closest_jamo(0.1, 0.5, 'vowel') # Bright/Dark vowel balance
        
        syllable = eng.synthesize_syllable(onset, nucleus)
        
        print(f"   [Math] Magnitude: {mag:.2f} | Roughness: {roughness:.2f} | Tension: {tension:.2f}")
        print(f"   [Linguistics] Closest Jamo: {onset} (Consonant), {nucleus} (Vowel)")
        print(f"   [Utterance] Elysia's sound: \"{syllable}\"")
        
        # 5. Narrative Explanation
        if 'A' in dna:
            print("   [Interpretation] Expansionary state detected. Soft consonants (ㄴ, ㄹ) emerge.")
        else:
            print("   [Interpretation] Contractive state detected. Tense/Hard consonants (ㄲ, ㄸ, ㅃ) emerge.")

if __name__ == "__main__":
    demonstrate_manifestation()
