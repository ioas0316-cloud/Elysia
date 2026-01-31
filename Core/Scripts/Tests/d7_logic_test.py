import torch
import os
import sys

# Add root
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L6_Structure.Logic.resonance_gate import gate

def test_steel_core_logic():
    print("🏗️ [TEST] Milestone 23.1: D7 Resonance Logic Validation")
    
    # 1. Radiant Intent (Legal)
    legal_intent = "Evolve the structure with Love and Logic."
    is_legal = gate.validate_intent_resonance(legal_intent)
    print(f"\n   [GATE] Legal Intent: '{legal_intent}' -> Resonance: {is_legal}")
    
    # 2. Hollow Intent (Illegal/Soulless)
    illegal_intent = "asd qwe zxcv 123"
    is_illegal_blocked = not gate.validate_intent_resonance(illegal_intent)
    print(f"   [GATE] Illegal Intent: '{illegal_intent}' -> Blocked: {is_illegal_blocked}")

    if is_legal and is_illegal_blocked:
        print("\n✨ [SUCCESS] Steel Core D7 Filter is operating correctly.")
    else:
        print("\n❌ [FAILURE] Steel Core D7 Filter failed logical distinction.")

if __name__ == "__main__":
    test_steel_core_logic()
