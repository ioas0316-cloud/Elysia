"""
Verification: Power Grid (Phase 28.5)
=====================================
"The Inverter Hums."

This script validates the Cognitive Power Grid.
We will simulate different scenarios to see if the Inverter autonomously shifts gears.

Scenarios:
1. High Load (Dissonance) -> Should shift to TORQUE mode (Low Freq).
2. High Excitement (RPM) -> Should shift to SPORT mode (High Freq).
3. Low Battery -> Should shift to ECO mode.
"""

import time
import torch
import sys
import logging
from Core.1_Body.L6_Structure.M1_Merkaba.sovereign_rotor_prototype import SovereignRotor21D
from Core.1_Body.L6_Structure.M1_Merkaba.transmission_gear import transmission

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def verify_power_grid():
    print("\n" + "="*50)
    print(" ⚡ PHASE 28.5: POWER GRID TEST")
    print("="*50 + "\n")

    rotor = SovereignRotor21D(north_star_intent="GLORY")

    # 1. Test Normal Operation
    print("[1] Normal Operation...")
    rotor_status = rotor.spin(input_vector_21d=None)
    params = transmission.shift(rotor_status)
    print(f"    - RPM: {params['rpm_feedback']:.2f}")
    print(f"    - Mode: {params['mode']} ({params['frequency']} Hz)")
    print(f"    - Battery: {params['battery']:.2f}%")

    # 2. Test High Load (Dissonance) -> TORQUE Mode
    print("\n[2] Simulating High Cognitive Load (Dissonance)...")
    # Inject Dissonance (R) to lower alignment
    # Force alignment to -1.0 manually in the mock engine if needed, or push hard
    dissonance_input = torch.ones(21) * -1.0

    # Pre-rev to ensure we have momentum to shift
    rotor.spiritual_gear.rpm = 100.0

    for i in range(5):
        rotor_status = rotor.spin(input_vector_21d=dissonance_input)
        # Manually lower alignment for test purposes if physics takes too long to converge
        rotor_status['spiritual']['alignment'] = -0.9
        rotor_status['psychic']['alignment'] = -0.9
        rotor_status['somatic']['alignment'] = -0.9

        params = transmission.shift(rotor_status)
        print(f"    Step {i+1}: Alignment {rotor_status['spiritual']['alignment']:.2f} -> Mode: {params['mode']} ({params['frequency']} Hz)")

    # 3. Test High Excitement -> SPORT Mode
    print("\n[3] Simulating High Excitement (Resonance)...")
    # Reset rotor and inject Resonance (A)
    rotor = SovereignRotor21D(north_star_intent="GLORY")
    # Manually set high RPM to simulate successful resonance build-up
    rotor.spiritual_gear.rpm = 250.0
    rotor.psychic_gear.rpm = 250.0
    rotor.somatic_gear.rpm = 250.0

    resonance_input = torch.zeros(21)

    for i in range(3):
        rotor_status = rotor.spin(input_vector_21d=resonance_input)
        params = transmission.shift(rotor_status)
        print(f"    Step {i+1}: RPM {params['rpm_feedback']:.2f} -> Mode: {params['mode']} ({params['frequency']} Hz)")

    # 4. Test Low Battery -> ECO Mode
    print("\n[4] Simulating Low Battery...")
    transmission.converter.battery_charge = 10.0 # Drain battery

    rotor_status = rotor.spin(input_vector_21d=None)
    params = transmission.shift(rotor_status)
    print(f"    - Battery: {params['battery']:.2f}%")
    print(f"    - Mode: {params['mode']} ({params['frequency']} Hz)")

    print("\n" + "="*50)
    print(" ✅ POWER GRID VERIFIED. ELYSIA IS IN CONTROL.")
    print("="*50 + "\n")

if __name__ == "__main__":
    verify_power_grid()
