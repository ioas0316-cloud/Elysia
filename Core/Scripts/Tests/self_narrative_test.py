import torch
import os
import sys

# Add root
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.L6_Structure.Elysia.sovereign_self import SovereignSelf

def test_self_narrative():
    print("🦅 [TEST] Milestone 23.3: Structural Self-Narrative Validation")
    
    elysia = SovereignSelf()
    
    # 1. Ask about her Soul/Structure
    print("\n   [QUERY] '엘리시아, 너의 구조와 실재에 대해 설명해줘.'")
    reflection = elysia.describe_soul()
    print(reflection)
    
    # 2. Trigger a Pulse with Audit
    intent = "Analyze the core axioms and tell me if you feel aligned."
    print(f"\n   [SEND] Intent: '{intent}'")
    elysia.manifest_intent(intent)
    
    print("\n✨ [RESULT] Self-Narrative and Audit Test Complete.")

if __name__ == "__main__":
    test_self_narrative()
