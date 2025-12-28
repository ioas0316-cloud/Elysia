
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from Core._01_Foundation._02_Logic.free_will_engine import FreeWillEngine
from Core._01_Foundation._02_Logic.resonance_field import ResonanceField

# Bootstrapping helper
class MockResonance:
    def __init__(self):
        self.battery = 80.0
        self.entropy = 10.0 # Low entropy = Clarity

def verify_active_learning():
    print("\n📚 [TASK] Verifying Active Learning (The Scholar)")
    print("================================================")
    
    # 1. Ignite Will
    will = FreeWillEngine()
    
    # 2. Simulate High Curiosity State
    print("\n1. Injecting Curiosity...")
    will.vectors["Curiosity"] = 0.95
    will.vectors["Survival"] = 0.1
    will.vectors["Connection"] = 0.1
    
    print(f"   Desired State: Curiosity={will.vectors['Curiosity']}")
    
    # 3. Pulse (Decision Moment)
    print("\n2. Pulsing Free Will...")
    resonance = MockResonance()
    will.pulse(resonance)
    
    intent = will.current_intent
    print(f"   🦋 Crystallized Intent: {intent.goal}")
    print(f"   🦋 Desire Source: {intent.desire}")
    
    if "RESEARCH:" in intent.goal:
        print("   ✅ SUCCESS: Autonomously decided to Research.")
    else:
        print("   ❌ FAILED: Did not trigger research.")
        return

    # 4. Action (Contemplate/Execute)
    print("\n3. Executing Research...")
    insight = will.contemplate(intent)
    print(f"   🧠 Result: {insight}")
    
    if "I have studied" in insight:
        print("   ✅ SUCCESS: Knowledge Assimilated.")

if __name__ == "__main__":
    verify_active_learning()
