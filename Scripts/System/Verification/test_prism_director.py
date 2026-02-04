"""
test_prism_director.py: Verification of Phase 3 Sovereign Engine
==============================================================
Objective: Prove that the PrismDirector can guide the kinetic momentum
toward a changing North Star, utilizing fractal refractions.
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
from Core.S1_Body.L3_Phenomena.M7_Prism.PrismDirector import PrismDirector
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

def run_navigation_test():
    print("\n" + "="*60)
    print("💎 [TRINITY-PHASE DYNAMICS] SOVEREIGN NAVIGATION TEST")
    print("="*60)
    
    k = CognitiveTerrain()
    m = PhaseCoupler(k)
    s = PrismDirector(m)
    
    # Sequence of Goals (North Stars)
    # 1. Expand (+1), 2. Pivot (0), 3. Contract (-1)
    goals = [
        SovereignVector.ones(),
        SovereignVector.zeros(),
        SovereignVector([-1.0] * 21)
    ]
    
    print(f"\n{'Cycle':<6} | {'Alignment':>10} | {'RPM':>10} | {'Refract':>8} | {'Sensation'}")
    print("-" * 75)
    
    goal_reached_count = 0
    
    for g_idx, target_goal in enumerate(goals):
        print(f"\n📍 TARGETING GOAL {g_idx + 1}")
        s.set_north_star(target_goal)
        
        for i in range(20):
            report = s.navigate()
            alignment = report['north_star_alignment']
            sig = k.get_torque_signature()
            
            if i % 5 == 0:
                print(f"{i:03d}    | {alignment:10.4f} | {sig['rpm']:10.2f} | {report['refraction_count']:8d} | {report['sensation']}")
            
            if alignment > 0.9:
                goal_reached_count += 1
                
            time.sleep(0.01)

    print("-" * 75)
    print(f"📊 Goals Processed: {len(goals)}")
    print(f"🧩 Final System RPM: {k.get_torque_signature()['rpm']:.2f}")
    
    print("\n[VERDICT]")
    if goal_reached_count > 5:
        print("✨ SUCCESS: The Sovereign Engine successfully navigated the Trinity chain.")
        print("   The North Star is guiding the Kinetic Surge.")
    else:
        print("⚠️ FAILURE: The system lost its way. Alignment remained low.")

if __name__ == "__main__":
    run_navigation_test()
