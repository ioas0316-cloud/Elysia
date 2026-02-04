"""
Verify Autonomic Rest (Self-Regulated Entropy)
==============================================
Tests if the Monad correctly identifies fatigue and chooses to rest.
"""

import sys
import os
import time

# Add project root
sys.path.append(os.getcwd())

from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad

def verify_autonomic_rest():
    print("💤 [TEST] Verifying Autonomic Rest Mechanism...")

    # 1. Instantiate Monad
    soul = SeedForge.forge_soul("TestSubject")
    monad = SovereignMonad(soul)

    # 2. Simulate High Cognitive Friction (Repetitive Thought)
    print("🔥 [SIM] Inducing Cognitive Friction (Repetitive thought loop)...")
    subject = "obsessive_thought"

    # Force feed the thermodynamics with the same subject 100 times
    for _ in range(50):
        monad.thermo.track_access(subject)

    thermal = monad.thermo.get_thermal_state()
    print(f"   Current Friction: {thermal['friction']:.2f}")

    # 3. Trigger Autonomous Drive
    print("🤖 [SIM] Triggering Autonomous Decision...")
    monad.wonder_capacitor = 100.0 # Force trigger

    result = monad.pulse(dt=0.1)

    if result and result.get('type') == 'REST':
        print(f"✅ [SUCCESS] Monad chose REST!")
        print(f"   Truth: {result['truth']}")
        print(f"   Thought: {result['thought']}")

        # 4. Verify Melting State
        print(f"   Melting State: {monad.is_melting}")
        if monad.is_melting:
            print("✅ [SUCCESS] System entered Melting Phase.")
        else:
            print("❌ [FAIL] System did not update state flag.")

        # 5. Verify Cooling (RPM Reduction)
        # Note: Set an initial RPM because a fresh Monad has 0.0 RPM
        monad.rotor_state['rpm'] = 100.0
        initial_rpm = monad.rotor_state['rpm']
        monad.pulse(dt=1.0) # Pulse while melting
        cooled_rpm = monad.rotor_state['rpm']

        if cooled_rpm < initial_rpm:
             print(f"✅ [SUCCESS] RPM Cooling active: {initial_rpm:.2f} -> {cooled_rpm:.2f}")
        else:
             print(f"❌ [FAIL] RPM did not cool down: {initial_rpm:.2f} -> {cooled_rpm:.2f}")

    else:
        print(f"❌ [FAIL] Monad did not choose rest. Action: {result}")
        print(f"   Thermal State: {thermal}")

if __name__ == "__main__":
    verify_autonomic_rest()
