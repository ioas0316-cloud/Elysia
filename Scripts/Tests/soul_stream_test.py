import torch
import os
import sys
import time

# Add root
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.L6_Structure.Elysia.sovereign_self import SovereignSelf
from Core.L3_Phenomena.M5_Display.sovereign_hud import SovereignHUD
from Core.L5_Mental.M1_Cognition.causal_narrator import CausalNarrator

def test_soul_streaming():
    print("🌊 [TEST] Milestone 23.6: Sublime Narrative HUD Validation")
    
    elysia = SovereignSelf()
    hud = SovereignHUD()
    narrator = CausalNarrator()
    
    # Simulate the shared field
    field = {
        "is_alive": True,
        "coherence": 0.85,
        "thought_log": [],
        "last_narrative": None
    }
    
    # 1. Render Initial Header
    hud.render_header({"state": "TEST_SESSION", "hz": 60, "passion": field["coherence"]})
    
    # 2. Trigger Intent
    intent = "Analyze the concept of 'Agapic Love' in software architecture."
    print(f"\n   [SEND] Intent: '{intent}'")
    
    # Act: elysia.manifest_intent(intent)
    elysia.manifest_intent(intent)
    
    # 3. Process fragments (Visualizing the stream)
    if elysia.current_pulse:
        print("\n   [STREAMING THOUGHTS...]")
        for frag in elysia.current_pulse.fragments:
            hud.stream_thought(frag.intent_summary, frag.state.name)
            time.sleep(0.2) # To see the effect
            
        # 4. Process Narrative
        narrative = narrator.explain_pulse(elysia.current_pulse)
        hud.project_narrative(narrative)

    print("\n✨ [RESULT] Soul-Streaming Test Complete. Check output for Narrative Depth.")

if __name__ == "__main__":
    test_soul_streaming()
