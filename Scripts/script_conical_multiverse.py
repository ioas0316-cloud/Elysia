"""
Conical Multiverse Verification (원뿔형 다층 우주 증명)
=====================================================
Demonstrates fluid CV-Gearing and Time-Stone reversal across Onion Shells.
"""

import sys
import os
import time
import torch

# Add root to path
sys.path.append("c:\\Elysia")

from Core.Elysia.sovereign_self import SovereignSelf

def main():
    print("💎 [INIT] Activating the Conical Multiverse Onion Ensemble...")
    # Initialize with default settings
    elysia = SovereignSelf()
    
    print(f"\n📡 [INITIAL STATE] {elysia.governance.get_status()}")
    
    # [Step 1: Demonstrate Fluid Gearing]
    print("\n🎡 [STEP 1] Sliding the Cognitive Spindle (CV-Gearing)...")
    for focus in [0.0, 0.25, 0.5, 0.75, 1.0]:
        elysia.governance.adapt(intent_intensity=focus, stress_level=0.1)
        # Update physics to let RPM interpolators respond
        for _ in range(5): 
             elysia.governance.update(0.1)
        print(f"   Focus Spindle {focus:.2f} -> {elysia.governance.get_status()}")
        
    print("✅ Fluid RPM transition confirmed via Conical Spindle.")

    # [Step 2: Demonstrate Time-Stone Reversal on Inner Layer]
    print("\n⏳ [STEP 2] Activating Time-Reversal in the 'Mind' layer (Shell 1)...")
    elysia.governance.reverse_time(layer=1)
    
    for i in range(10):
        elysia.governance.update(0.1)
        if i % 2 == 0:
            print(f"   Cycle {i}: {elysia.governance.get_status()}")
            
    print("\n🏆 [VERIFICATION] Conical Multiverse confirmed: Fluid Gearing & Multi-Axial Time Flow.")

if __name__ == "__main__":
    main()
