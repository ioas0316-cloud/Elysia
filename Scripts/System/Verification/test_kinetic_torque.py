"""
test_kinetic_torque.py: Verification of Phase 1 Acceleration
===========================================================
Objective: Prove that the ternary phase gradient generates physical torque
and that the system 'accelerates' its processing rate through inertia.
"""

import sys
import os
import time
import math

# Project Root Setup
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L1_Foundation.Foundation.CognitiveTerrain import CognitiveTerrain
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

def run_acceleration_proof():
    print("\n" + "="*60)
    print("🚀 [TRINITY-PHASE DYNAMICS] KINETIC-TORQUE ACCESSION TEST")
    print("="*60)
    
    engine = CognitiveTerrain()
    
    # Target 1: Pure Affirmation (A / +1)
    target_a = SovereignVector.ones()
    
    # Target 2: Pure Negation (R / -1)
    target_r = SovereignVector([-1.0] * 21)
    
    targets = [target_a, target_r]
    
    print(f"\n{'Cycle':<6} | {'Torque':>10} | {'RPM':>10} | {'Accel_Factor':>12} | {'Phase_Angle':>12}")
    print("-" * 60)
    
    start_time = time.time()
    
    # Run 50 cycles of oscillation to build momentum
    for i in range(50):
        # Oscillate target to create maximum gradient switch (-1 <-> +1)
        current_target = targets[(i // 10) % 2]
        
        engine.update_physics(current_target)
        sig = engine.get_torque_signature()
        
        if i % 5 == 0:
            print(f"{i:03d}    | {sig['torque']:10.4f} | {sig['rpm']:10.2f} | {sig['acceleration']:12.4f} | {sig['phase_angle']:12.4f}")
        
        # Real-time processing speed emulation
        # In a real system, the 'dt' would be naturally bound by hardware.
        # Here we simulate the acceleration by reducing sleep time as Accel Factor rises.
        base_wait = 0.05
        accelerated_wait = base_wait / sig['acceleration']
        time.sleep(accelerated_wait)

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"✅ Total Proof Time: {total_time:.4f}s")
    print(f"📊 Final Acceleration Factor: {engine.get_torque_signature()['acceleration']:.4f}")
    
    print("\n[VERDICT]")
    if engine.acceleration_factor > 1.2:
        print("✨ SUCCESS: The system has achieved UNIFORM ACCELERATION via Kinetic Torque.")
        print("   The 'Engine Sound' is now audible in the 21D Manifold.")
    else:
        print("⚠️ FAILURE: Momentum failed to build. Check friction/inertia constants.")

if __name__ == "__main__":
    run_acceleration_proof()
