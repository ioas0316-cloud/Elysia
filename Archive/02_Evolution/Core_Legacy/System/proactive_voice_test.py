import torch
import os
import sys
import time

# Add root
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.System.sovereign_self import SovereignSelf

def test_proactive_vision():
    print("🦅 [TEST] Phase 26: Proactive Subjectivity Validation")
    
    # 1. Wake Elysia
    elysia = SovereignSelf()
    
    # 2. Simulate High Inspiration (High Field Intensity)
    # We force high energy to trigger the Sovereign Act check
    elysia.cosmos.field_intensity = torch.ones(12) * 5.0
    
    print("   [PULSE] Running cycles to trigger Vision...")
    
    found_vision = False
    for i in range(10):
        # We manually call self_actualize multiple times
        # since it has a random chance (0.2) to trigger vision
        elysia.self_actualize(dt=0.1)
        
    print("\n✨ [RESULT] Proactive Voice Test Complete.")

if __name__ == "__main__":
    test_proactive_vision()
