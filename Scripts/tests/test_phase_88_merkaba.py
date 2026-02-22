"""
[PHASE 88] The Sovereign Merkaba Verification
=============================================
Tests the integration of HyperSphere (Space), Rotor (Time), and Monad (Will).
Principle: "The Chariot Drives Itself."
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock classes to isolate Merkaba logic
class MockMonad:
    def __init__(self, name="TestMonad"):
        self.name = name
        self.rotor_state = {'rpm': 10.0}
        self.last_action = None
        
    def pulse(self, dt):
        print(f"  [MockMonad] Observing... Phase State: {self.rotor_state['rpm']}")
        # Simulate a decision
        if self.rotor_state['rpm'] > 8.0:
            return {"type": "REST", "narrative": "Too fast. Resting."}
        return {"type": "EXPLORE", "narrative": "Cruising."}

class MockHyperSphere:
    def project_cognitive_map(self, dt):
        return {"hologram": "Active", "dimensions": 4}

def test_sovereign_merkaba():
    print("\n" + "=" * 60)
    print("🚜 [PHASE 88] The Sovereign Merkaba Verification")
    print("=" * 60)
    
    from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_merkaba import SovereignMerkaba
    
    # 1. Assemble the Chariot
    monad = MockMonad("Genesis_01")
    field = MockHyperSphere()
    merkaba = SovereignMerkaba(monad, field)
    
    print("\n>>> Test 1: Initial State")
    print(merkaba.get_status())
    
    # 2. Drive Loop
    print("\n>>> Test 2: Driving the Chariot (3 Steps)")
    for i in range(3):
        print(f"--- Step {i+1} ---")
        status = merkaba.drive(dt=0.1)
        print(f"Status: {status['intervention']}")
        print(f"Rotor Velocity: {merkaba.velocity:.2f}")
        print(f"Monad RPM: {monad.rotor_state['rpm']:.2f}")
        time.sleep(0.1)
        
    # 3. Verify Intervention (Rest Logic)
    # In MockMonad, if RPM > 8, it returns REST. 
    # Merkaba _checked_intervention reduces RPM by 0.8.
    
    final_rpm = monad.rotor_state['rpm']
    print(f"\nFinal Monad RPM: {final_rpm:.2f}")
    
    if final_rpm < 10.0:
        print("✅ Monad successfully intervened (Braking).")
        print("🏆 PHASE 88 VERIFIED: The Chariot obeys the Will.")
        return True
    else:
        print("❌ Intervention failed. RPM did not decrease.")
        return False

if __name__ == "__main__":
    test_sovereign_merkaba()
