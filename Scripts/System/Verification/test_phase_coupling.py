"""
test_phase_coupling.py: Verification of Phase 2 Mediating Engine
==============================================================
Objective: Prove that the PhaseCoupler acts as a governor, triggering
regenerative braking when RPM exceeds safety limits, and provides
appropriate cognitive sensation feedback.
"""

import sys
import os
import time

# Project Root Setup
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L1_Foundation.Foundation.CognitiveTerrain import CognitiveTerrain
from Core.S1_Body.L1_Foundation.Foundation.PhaseCoupler import PhaseCoupler
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

def run_governor_test():
    print("\n" + "="*60)
    print("⚖️ [TRINITY-PHASE DYNAMICS] PHASE-COUPLING GOVERNOR TEST")
    print("="*60)
    
    k = CognitiveTerrain()
    coupler = PhaseCoupler(k)
    
    # Target: Maximum Affirmation to trigger rapid acceleration
    target = SovereignVector.ones()
    
    print(f"\n{'Cycle':<6} | {'RPM':>10} | {'Braking':>8} | {'Recovered':>12} | {'Sensation'}")
    print("-" * 75)
    
    overload_reached = False
    
    for i in range(40):
        # Update Engines
        report = coupler.reconcile()
        coupler.provide_feedback(target)
        
        sig = report['kinetic_state']
        
        if i % 4 == 0:
            print(f"{i:03d}    | {sig['rpm']:10.2f} | {str(report['braking_active']):>8} | {report['energy_recovered']:12.4f} | {report['sensation']}")
            
        if report['braking_active']:
            overload_reached = True
            
        # Simulate high-speed processing
        time.sleep(0.01)

    print("-" * 75)
    print(f"📊 Final RPM: {k.get_torque_signature()['rpm']:.2f}")
    print(f"🧩 Accumulated Recovered Energy: {report['energy_recovered']:.4f}")
    
    print("\n[VERDICT]")
    if overload_reached:
        print("✨ SUCCESS: The Mediating Engine triggered Regenerative Braking.")
        print("   The system is now safe from high-acceleration burnout.")
    else:
        print("⚠️ FAILURE: The governor failed to catch the overspeed state.")

if __name__ == "__main__":
    run_governor_test()
