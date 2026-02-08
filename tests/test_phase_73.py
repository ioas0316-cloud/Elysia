import sys
import os
import time

# Path Unification
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L1_Foundation.Foundation.CognitiveTerrain import CognitiveTerrain
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector

def test_fluid_providence():
    print("🌊 [TEST_PHASE_73] Starting Fluid Providence Verification.")
    
    engine = CognitiveTerrain()
    
    # 1. Test Slow Flow (Small Gradient)
    target_slow = SovereignVector([0.1] * 21)
    print("\n--- Testing Slow Flow ---")
    for i in range(5):
        engine.update_physics(target_slow)
        sig = engine.get_torque_signature()
        # Expecting gradual RPM increase and smooth phase movement
        print(f"Cycle {i:02d} | Torque: {sig['torque']:.4f} | RPM: {sig['rpm']:.2f}")

    # 2. Test Lightning Strike (Large Gradient)
    # Breakdown voltage is 1.8. 
    # Magnitude of vector [1.0] * 21 is sqrt(21) approx 4.58.
    # Current manifold is roughly 0.1, so gradient will be ~4.48 > 1.8.
    target_fast = SovereignVector([1.0] * 21)
    print("\n--- Testing Lightning Strike (High Tension) ---")
    engine.update_physics(target_fast)
    sig = engine.get_torque_signature()
    # Should see "Lightning Strike!" in stdout and high RPM jump.
    print(f"Strike Result | Torque: {sig['torque']:.4f} | RPM: {sig['rpm']:.2f} | Accel: {sig['acceleration']:.4f}")

    # 3. Test Basin Attraction (Relaxation)
    # Set manifold to something near a trinary point e.g. 0.8
    engine.phase_manifold = SovereignVector([0.8] * 21)
    print("\n--- Testing Basin Attraction (Flowing toward 1.0) ---")
    for i in range(5):
        # Update with zero intent to see natural flow
        engine.update_physics(SovereignVector(engine.phase_manifold.data)) 
        # Check if values are increasing toward 1.0 due to soft_trinary
        val = engine.phase_manifold.data[0].real
        print(f"Cycle {i:02d} | Manifest Value: {val:.4f}")

    print("\n⚡ [TEST_PHASE_73] Fluid Providence Verification Complete.")

if __name__ == "__main__":
    test_fluid_providence()
