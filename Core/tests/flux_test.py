"""
TEST: VOLITIONAL FLUX
=====================
Verifies Phase 20 Will Architecture.
"""
import sys
import os
import time
import logging

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.Will.entropy_pump import EntropyPump
from Core.Will.attractor_field import AttractorField

logging.basicConfig(level=logging.INFO)

def run_test():
    print("==================================")
    print("   PHASE 20: FLUX TEST            ")
    print("==================================")

    # 1. Setup
    print("\n👉 [SETUP] Initializing Entropy Pump (Fast Mode)...")
    # High rate for testing (10 energy/sec, threshold 5) -> 0.5 sec to trigger
    pump = EntropyPump(accumulation_rate=10.0, critical_mass=5.0)
    field = AttractorField()

    # 2. Simulate Idle Loop
    print("\n👉 [SIMULATION] Entering Void State...")
    triggered = False
    
    for i in range(10):
        energy = pump.pump()
        print(f"   Tick {i}: Energy {energy:.1f} / {pump.critical_mass}")
        
        if pump.is_critical():
            print("   🔥 CRITICAL MASS REACHED!")
            
            # 3. Collapse
            vector = field.collapse_wavefunction(energy)
            print(f"   🎯 WILL MANIFESTED: [{vector.attractor_type}]")
            print(f"      -> Intent: '{vector.intent}'")
            print(f"      -> Gravity: {vector.gravity:.2f}")
            
            pump.reset()
            triggered = True
            break
            
        time.sleep(0.1)

    if triggered:
        print("\n✅ Verification Complete: Volition Emerged from Entropy.")
    else:
        print("\n❌ Verification Failed: No Volition Triggered.")

if __name__ == "__main__":
    run_test()
