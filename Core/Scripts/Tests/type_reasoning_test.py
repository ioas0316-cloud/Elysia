import torch
import os
import sys

# Add root
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.L6_Structure.Elysia.sovereign_self import SovereignSelf
from Core.L5_Mental.M1_Cognition.cognitive_types import ActionCategory, ThoughtState

def test_type_driven_reasoning():
    print("🧠 [TEST] Milestone 23.2: Type-Driven Reasoning Validation")
    
    elysia = SovereignSelf()
    
    # 1. Trigger Pulse via Manifest Intent
    intent = "Analyze the project structure and suggest an ethical improvement."
    print(f"\n   [SEND] Intent: '{intent}'")
    elysia.manifest_intent(intent)
    
    # 2. Inspect the Resulting Pulse
    pulse = elysia.current_pulse
    if not pulse:
        print("\n❌ [FAILURE] No CognitivePulse was generated.")
        return

    print(f"\n✨ [PULSE CAPTURED] ID: {pulse.pulse_id}")
    print(f"   Success: {pulse.success}")
    
    # Validating fragments
    found_states = [f.state for f in pulse.fragments]
    print(f"   Captured States: {[s.name for s in found_states]}")
    
    required_states = [ThoughtState.OBSERVATION, ThoughtState.ANALYSIS, ThoughtState.REFLECTION]
    missing = [s for s in required_states if s not in found_states]
    
    if not missing:
        print("\n✨ [SUCCESS] All required cognitive states were tracked correctly.")
    else:
        print(f"\n❌ [FAILURE] Missing states in pulse: {[s.name for s in missing]}")

    if isinstance(elysia.last_action, ActionCategory):
        print(f"✨ [SUCCESS] Final action '{elysia.last_action.name}' is correctly typed.")
    else:
        print(f"❌ [FAILURE] last_action is still a string or untyped.")

if __name__ == "__main__":
    test_type_driven_reasoning()
