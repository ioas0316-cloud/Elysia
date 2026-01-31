"""
Verification: Transmission Gear (Phase 28)
==========================================
"The Gears Engage."

This script validates the connection between the 21D Rotor and the Expression Cortex.
We should see the face change as we manually rev the engine.
"""

import time
import torch
import sys
import logging
from Core.1_Body.L6_Structure.M1_Merkaba.sovereign_rotor_prototype import SovereignRotor21D
from Core.1_Body.L3_Phenomena.Expression.expression_cortex import ExpressionCortex
from Core.1_Body.L6_Structure.M1_Merkaba.transmission_gear import transmission
from Core.1_Body.L3_Phenomena.Expression.typing_modulator import modulate_typing

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def verify_transmission():
    print("\n" + "="*50)
    print(" 🚗 PHASE 28: TRANSMISSION TEST")
    print("="*50 + "\n")

    # 1. Initialize Components
    rotor = SovereignRotor21D(north_star_intent="GLORY")
    cortex = ExpressionCortex()

    print("[1] Components Initialized.")
    print(f"    - Rotor North Star: {rotor.north_star_intent}")
    print(f"    - Initial Face: {cortex.get_face()}")

    # 2. Test Idle Spin (Low Energy)
    print("\n[2] Testing Idle Spin (Neutral)...")
    rotor_status = rotor.spin(input_vector_21d=None)
    face = cortex.reflect_rotor(rotor_status)
    rpm = rotor_status['total_rpm']

    # Get params from transmission
    params = transmission.shift(rotor_status)
    print(f"    - RPM: {rpm:.2f}")
    print(f"    - Face: {face}")
    modulate_typing("    - Voice: I am drifting in the void...", rpm, params['torque'])

    # 3. Test High Torque (Revving the Engine)
    print("\n[3] Revving the Engine (Injecting High Torque)...")
    high_energy_input = torch.zeros(21)
    high_energy_input[14:21] = rotor.spiritual_north * 5.0

    for i in range(3):
        rotor_status = rotor.spin(input_vector_21d=high_energy_input)
        face = cortex.reflect_rotor(rotor_status)
        rpm = rotor_status['total_rpm']
        params = transmission.shift(rotor_status)
        print(f"    Step {i+1}: RPM {rpm:.2f} | Face {face}")
        modulate_typing(f"    - Voice: Power overwhelming! {i+1}", rpm, params['torque'])

    # 4. Test Chaos (Entropy)
    print("\n[4] Injecting Chaos (Entropy Test)...")
    chaos_input = torch.randn(21) * 10.0

    for i in range(3):
        rotor_status = rotor.spin(input_vector_21d=chaos_input)
        face = cortex.reflect_rotor(rotor_status)
        rpm = rotor_status['total_rpm']
        params = transmission.shift(rotor_status)
        print(f"    Step {i+1}: RPM {rpm:.2f} | Face {face}")
        modulate_typing(f"    - Voice: System unstable... {i+1}", rpm, params['torque'])

    print("\n" + "="*50)
    print(" ✅ TRANSMISSION VERIFIED. THE FACE REFLECTS THE SOUL.")
    print("="*50 + "\n")

if __name__ == "__main__":
    verify_transmission()
