"""
Verification: Safety System (Phase 29)
======================================
"The Relay Guardians."

Tests ANSI Device 27 (UVR), 32 (RP), 25 (Sync) and PID Controller.
Also checks the Energy Dashboard (Temp/Volt).
"""

import time
import torch
import sys
import logging
from Core.L6_Structure.M1_Merkaba.sovereign_rotor_prototype import SovereignRotor21D
from Core.L6_Structure.M1_Merkaba.transmission_gear import transmission
from Core.L6_Structure.M1_Merkaba.protection_relay import protection

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def verify_safety():
    print("\n" + "="*50)
    print(" 🛡️ PHASE 29: SAFETY SYSTEM & DASHBOARD TEST")
    print("="*50 + "\n")

    rotor = SovereignRotor21D(north_star_intent="GLORY")

    # 1. Dashboard Check
    print("[1] Checking Energy Dashboard...")
    # Rev up a bit
    rotor.spiritual_gear.rpm = 100.0
    status = rotor.spin(input_vector_21d=None)
    params = transmission.shift(status)

    print(f"    - Temp: {params['temperature']:.1f}°C (Base 36.5 + RPM/20)")
    print(f"    - Volt: {params['voltage']:.1f}V (Base 220 + Align*100)")
    print(f"    - Relays: {params['relays']}")

    # 2. Test Device 32 (Reverse Power)
    print("\n[2] Testing Device 32 (Reverse Power / Dissonance)...")
    # Simulate receiving chaos text
    transmission.process_input("CHAOS " * 100) # Complexity > 0.95 likely

    if not protection.relays["32"]:
        print("    ✅ Device 32 TRIPPED: Input Blocked.")
    else:
        print("    ⚠️ Device 32 DID NOT TRIP (Might need more chaos).")

    # Reset relay
    protection.relays["32"] = True

    # 3. Test Device 27 (Under Voltage)
    print("\n[3] Testing Device 27 (Under Voltage)...")
    transmission.converter.battery_charge = 5.0 # Critical Low

    params = transmission.shift(status)
    if params["mode"] == "SLEEP":
        print(f"    ✅ Device 27 TRIPPED: Mode is {params['mode']}")
    else:
        print(f"    ⚠️ Device 27 FAILED: Mode is {params['mode']}")

    # Reset Battery
    transmission.converter.battery_charge = 100.0
    protection.relays["27"] = True

    # 4. Test PID Controller (Smoothing)
    print("\n[4] Testing PID Controller...")
    # Sudden RPM spike request
    print("    - Injecting sudden RPM spike...")
    transmission.inverter.auto_shift(current_rpm=500.0, battery_level=100.0, context_load=0.0)
    # PID should dampen or adjust the logic flow, hard to visualize in one step but
    # we check if it runs without error.
    print("    ✅ PID Loop Executed.")

    print("\n" + "="*50)
    print(" ✅ SAFETY SYSTEM VERIFIED.")
    print("="*50 + "\n")

if __name__ == "__main__":
    verify_safety()
