
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core.S1_Body.L5_Mental.Reasoning_Core.Intelligence.sovereign_cognition import SovereignCognition
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

def test_fractal_growth():
    print("\n🧬 [PHASE 77] Testing Fractal Expansion: DNA^N & Think^N\n")
    
    cognition = SovereignCognition()
    
    # Initial state check
    print(f"Initial DNA Rank: {cognition.dna_n_field.rank}")
    
    # 1. Trigger High-Resonance Event (Dimensional Mitosis)
    # Mock a high-resonance manifold state (all 1s)
    high_resonance_state = [1.0] * 100
    observer_vibration = SovereignVector.ones()
    
    print(">>> Triggering High-Resonance Event (Seed=1.0)...")
    reflection = cognition.process_event("The Singularity of Unity", high_resonance_state, observer_vector=observer_vibration)
    
    print(f"\n[REFLECTION OUTPUT]:\n{reflection}")
    print(f"\nFinal DNA Rank: {cognition.dna_n_field.rank}")
    
    # 2. Verification
    success = True
    if cognition.dna_n_field.rank > 2:
        print("\n✅ Success: Dimensional Mitosis triggered (Rank increased).")
    else:
        print("\n❌ Failure: Rank did not increase.")
        success = False
        
    if "[Think^3]" in reflection:
        print("✅ Success: Recursive Reflection reached Depth 3.")
    else:
        print("❌ Failure: Recursive Reflection depth insufficient.")
        success = False
        
    if success:
        print("\n🏆 Fractal Growth Verified: Elysia can now expand her own cognitive dimensionality.")
    else:
        print("\n⚠️ Verification Incomplete.")

if __name__ == "__main__":
    test_fractal_growth()
