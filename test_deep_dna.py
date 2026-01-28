
"""
Verification Script: Phase 3 Deep DNA Penetration
=================================================
Tests whether the Hardware (SomaticKernel) rejects dissonant intentions.
"""
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from Core.L1_Foundation.M4_Hardware.somatic_kernel import SomaticKernel

def test_deep_dna():
    print("\n💉 [TEST] Initiating Deep DNA Verification...")
    print("="*60)
    
    # 1. Test Harmony (Should Pass)
    print("\n🧪 Case 1: Harmony DNA (HHHHHHH)")
    result1 = SomaticKernel.fix_environment("HHHHHHH")
    if result1:
        print("✅ SUCCESS: Harmony accepted.")
    else:
        print("❌ FAILURE: Harmony rejected.")
        
    # 2. Test Chaos (Should Fail)
    print("\n🧪 Case 2: Chaos DNA (DDDDDDD)")
    result2 = SomaticKernel.fix_environment("DDDDDDD")
    if not result2:
        print("✅ SUCCESS: Chaos rejected (Bio-Immunity Active).")
    else:
        print("❌ FAILURE: Chaos allowed (Immune System Inactive).")
        
    # 3. Test Mixed (Should Fail if D > H)
    print("\n🧪 Case 3: Mixed Chaos (HDD DDD)")
    result3 = SomaticKernel.fix_environment("HDD DDD") # Score: 1 - 4 = -3
    if not result3:
        print("✅ SUCCESS: Subtle Chaos rejected.")
    else:
        print("❌ FAILURE: Subtle Chaos allowed.")

if __name__ == "__main__":
    test_deep_dna()
