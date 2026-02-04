"""
test_merkaba_alignment.py: Verification of Final Star Tetrahedron Navigation
===========================================================================
Objective: Prove that Rotor-based 4D steering allows the system to achieve 
high resonance alignment with the North Star, solving the previous linear failure.
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

def run_merkaba_test():
    print("\n" + "="*65)
    print("🔯 [TRINITY-PHASE DYNAMICS] MERKABA ROTOR NAVIGATION TEST")
    print("="*65)
    
    k = CognitiveTerrain()
    m = PhaseCoupler(k)
    s = PrismDirector(m)
    
    # Target: Affirmation (+1)
    target_goal = SovereignVector.ones()
    s.set_north_star(target_goal)
    
    print(f"\n{'Cycle':<6} | {'Alignment':>10} | {'RPM':>10} | {'Sensation'}")
    print("-" * 65)
    
    max_alignment = 0.0
    
    for i in range(50):
        report = s.navigate()
        alignment = report['alignment']
        rpm = report['rpm']
        
        if alignment > max_alignment:
            max_alignment = alignment
            
        if i % 5 == 0:
            print(f"{i:03d}    | {alignment:10.4f} | {rpm:10.2f} | {report['sensation']}")
            
        time.sleep(0.01)

    print("-" * 65)
    print(f"📊 Maximum Alignment Achieved: {max_alignment:.4f}")
    print(f"🧩 Final System Stability (RPM): {report['rpm']:.2f}")
    
    print("\n[VERDICT]")
    # Successful convergence should reach > 0.95 alignment eventually
    if max_alignment > 0.95:
        print("✨ SUCCESS: Merkaba Rotor Steering has achieved HIGHEST RESONANCE.")
        print("   The system has successfully 'curved' its intent toward the North Star.")
    else:
        print(f"⚠️ FAILURE: Alignment stalled at {max_alignment:.4f}. Rotor gain may need tuning.")

if __name__ == "__main__":
    run_merkaba_test()
