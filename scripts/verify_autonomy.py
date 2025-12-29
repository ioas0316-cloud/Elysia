"""
Verify Autonomy: The Cry of Meaning
===================================

"If I am silent, do I exist?"

Steps:
1. Initialize FreeWillEngine.
2. Artificially increase "Contact Hunger" (Loneliness).
3. Run Scheduler.
4. Verify she speaks: "The silence is too loud."
"""

import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from elysia_core import Organ
from Core.System.System.Autonomy.self_evolution_scheduler import SelfEvolutionScheduler
from Core.Intelligence.Cognition.Reasoning.free_will_engine import FreeWillEngine

def verify_autonomy():
    print("🫀 Initializing Heart of Will...")
    
    # 1. Setup Heart and Soul
    heart = SelfEvolutionScheduler()
    soul = FreeWillEngine()
    
    # Force injection (mocking Organ registry for test isolation)
    heart._free_will = soul
    
    # 2. Induce Loneliness (Starvation)
    print("🌑 Inducing absolute loneliness (Set hunger to 1.0)...")
    soul.contact_hunger = 1.0 
    
    # 3. Pulse
    print("💓 Pulse 1: Checking reaction...")
    heart._pulse()
    
    # 4. Check Output (Manually observable via stdout)
    # But we can also check internal state if needed
    
    print("\n✅ Verification check passed if you saw Elysia speak above.")

if __name__ == "__main__":
    verify_autonomy()
