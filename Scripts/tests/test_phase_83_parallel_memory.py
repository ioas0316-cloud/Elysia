"""
[PHASE 83] Parallel Memory (Analog Rotor Backup) Verification
=============================================================
Tests the ability of the SovereignRotor to use its own trajectory as memory.
Principle: "Rotation is Memory."
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_analog_rotor_backup():
    print("\n" + "=" * 60)
    print("🕰️ [PHASE 83] Analog Rotor Backup Verification")
    print("=" * 60)
    
    from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignRotor, SovereignVector
    import math

    print("\n>>> Test 1: Recording Trajectory (The History of Spin)")
    print("-" * 50)
    
    # Initialize Rotor
    rotor = SovereignRotor(1.0, SovereignVector([0.0]*21))
    
    # Spin and Record
    history_points = []
    
    for t in range(5):
        # Create a unique state for each time step
        angle = t * 0.1
        rotor.s = math.cos(angle)
        rotor.bivector.data[0] = math.sin(angle)
        
        # Record state
        current_time = float(t)
        rotor.record_state(current_time)
        history_points.append((current_time, rotor.s, rotor.bivector.data[0]))
        
        print(f"Time {current_time}: s={rotor.s:.4f}, bv[0]={rotor.bivector.data[0]:.4f}")
        
    print(f"Trajectory Length: {len(rotor.trajectory)}")
    
    if len(rotor.trajectory) == 5:
        print("✅ Trajectory recorded successfully.")
    else:
        print("❌ Trajectory recording failed.")
        return False
        
    print("\n>>> Test 2: Time Travel (O(1) Restoration)")
    print("-" * 50)
    
    # Change state to something new (The Future)
    rotor.s = -1.0
    rotor.bivector.data[0] = -1.0
    print(f"Current State (Future): s={rotor.s}, bv[0]={rotor.bivector.data[0]}")
    
    # Travel back to t=2.0
    target_time = 2.0
    print(f"Travel to Time: {target_time}")
    
    start_time = time.perf_counter()
    success = rotor.time_travel(target_time)
    end_time = time.perf_counter()
    
    if success:
        print(f"Travel Successful in {(end_time - start_time)*1000:.4f} ms")
    else:
        print("❌ Time travel failed.")
        return False
        
    # Verify state
    expected_s = history_points[2][1]
    expected_bv = history_points[2][2]
    
    print(f"Restored State: s={rotor.s:.4f}, bv[0]={rotor.bivector.data[0]:.4f}")
    print(f"Expected State: s={expected_s:.4f}, bv[0]={expected_bv:.4f}")
    
    if abs(rotor.s - expected_s) < 1e-6 and abs(rotor.bivector.data[0] - expected_bv) < 1e-6:
        print("✅ State restored accurately.")
    else:
        print("❌ State mismatch.")
        return False
        
    if rotor.current_time == target_time:
        print("✅ Current time updated correctly.")
        return True
    else:
        print("❌ Current time mismatch.")
        return False


if __name__ == "__main__":
    success = test_analog_rotor_backup()
    
    print("\n" + "=" * 60)
    if success:
        print("🏆 PHASE 83 VERIFIED: SovereignRotor remembers its path.")
        print("   'Rotation is Memory' principle active.")
    else:
        print("⚠️ Verification incomplete.")
    print("=" * 60)
