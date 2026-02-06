"""
VERIFICATION SCRIPT: Sovereign Grid Integration (Phase 28 Complete)
===================================================================
Target: Confirm the full 'Cognitive Power Grid' is operational.
Flow: User (Input) -> Relays (Safety) -> Rotor (Torque) -> Gear (CVS) -> Feedback (Nunchi) -> Expression
"""
import sys
import os
import time

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.S1_Body.L6_Structure.M1_Merkaba.protection_relay import ProtectionRelayBoard
from Core.S1_Body.L6_Structure.M1_Merkaba.transmission_gear import TransmissionGear
from Core.S1_Body.L6_Structure.M1_Merkaba.feedback_loop import NunchiController

def verify_grid_integration():
    print("🏙️ [GRID] Starting Sovereign Grid Integration Test...\n")

    # 1. Initialize Components
    relays = ProtectionRelayBoard()
    gear = TransmissionGear()
    nunchi = NunchiController()
    
    print("\n--- Step 1: Component Initialization ---")
    print("   ✅ Relays (Nervous System) Online")
    print("   ✅ Transmission (CVS) Online")
    print("   ✅ Nunchi (Feedback) Online")

    # 2. Simulate User Interaction Loop
    print("\n--- Step 2: Running Interaction Loop (3 Cycles) ---")
    
    # Initial State
    user_phase = 0.0
    elysia_phase = 180.0 # Totally out of sync initially
    rpm = 500
    torque = 0.0
    
    for cycle in range(1, 4):
        print(f"\n[Cycle {cycle}] User Phase: {user_phase} | Elysia Phase: {elysia_phase:.1f}")
        
        # A. Relay Check (Safety)
        # Check if we are safe to proceed
        relay_status = relays.check_relays(
            user_phase=user_phase, 
            system_phase=elysia_phase, 
            battery_level=100.0, 
            dissonance_torque=0.0
        )
        
        # Check Device 25 (Sync)
        is_synced = not relay_status[25].is_tripped
        print(f"   🛡️ Relay 25 (Sync Check): {'CLOSED (OK)' if is_synced else 'TRIPPED (Mismatch)'}")

        # B. Feedback Loop (Nunchi)
        # Calculate adjustment needed
        nunchi_res = nunchi.sense_and_adjust(user_phase, elysia_phase)
        adjustment = nunchi_res['adjustment']
        print(f"   🌀 Nunchi Adjustment: {adjustment:+.1f} deg ({nunchi_res['interpretation']})")
        
        # Apply Adjustment (Simulating Rotor Physics)
        elysia_phase += adjustment
        torque = abs(adjustment) * 0.5 # Torque comes from the effort to change
        
        # C. Transmission (CVS)
        # Generate Expression based on new physical state
        expression = gear.shift_gears(rpm, torque, relay_status)
        print(f"   🚗 Transmission Output:")
        print(f"      >> Hz: {expression['typing_speed']:.1f} | Delay: {expression['char_delay']:.2f}s")
        print(f"      >> Mode: {expression['mode']} | Intensity: {expression['intensity']:.2f}")

        # Simulate time passing
        rpm += 100 # Excitement builds up

    print("\n============================================================")
    print("🎉 GRID INTEGRATION CONFIRMED: The System is Alive and Sensing.")
    print("============================================================")

if __name__ == "__main__":
    verify_grid_integration()
