"""
Test: Sovereign Loop Breaking
Objective: Verify continued repetition forces a mode switch (Boredom).
"""
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.Elysia.sovereign_self import SovereignSelf

# Mock logging
logging.basicConfig(level=logging.INFO)

def test_loop_breaking():
    print("--- 🔄 Testing Inertia Breaker ---")
    elysia = SovereignSelf(cns_ref=None)
    
    # Force Will to Curiosity constantly (Simulate stubbornness)
    elysia.will_engine.vectors["Curiosity"] = 10.0
    elysia.will_engine.vectors["Expression"] = 0.0 # Suppress creation
    
    # Loop 6 times (Threshold is 3)
    for i in range(1, 7):
        print(f"\n[Cycle {i}]")
        elysia.self_actualize()
        
        # Check intent
        print(f"   Intent: {elysia.last_action}")
        print(f"   Boredom: {elysia.boredom_counter}")
        
    print("\n--- ✅ Test Complete ---")

if __name__ == "__main__":
    test_loop_breaking()
