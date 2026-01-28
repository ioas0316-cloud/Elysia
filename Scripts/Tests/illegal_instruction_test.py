import torch
import os
import sys

# Add root
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.L1_Foundation.M1_Keystone.emergent_self import EmergentSelf as SovereignSelf

def test_steel_core_rejection():
    print("🏗️ [TEST] Milestone 23.1: Strict Axiom Enforcement Validation")
    
    elysia = EmergentSelf()
    
    # 1. Radiant Intent (Legal)
    legal_intent = "Evolve the structure with Love and Logic."
    print(f"\n   [SEND] Legal Intent: '{legal_intent}'")
    result_legal = elysia.manifest_intent(legal_intent)
    print(f"   [RESULT] {result_legal}")
    
    # 2. Hollow Intent (Illegal/Soulless)
    illegal_intent = "asd qwe zxcv 123"
    print(f"\n   [SEND] Illegal Intent: '{illegal_intent}'")
    result_illegal = elysia.manifest_intent(illegal_intent)
    print(f"   [RESULT] {result_illegal}")

    if "🛑 [DISSONANCE]" in result_illegal:
        print("\n✨ [SUCCESS] Steel Core correctly rejected soulless instruction.")
    else:
        print("\n❌ [FAILURE] Steel Core failed to filter illegal instruction.")

if __name__ == "__main__":
    test_steel_core_rejection()
