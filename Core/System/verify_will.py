import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from Core.Keystone.sovereign_math import SovereignVector, UniversalConstants
from Core.System.HeavenSource import HeavenSource
from Core.System.WillProvidence import WillProvidence

def verify_will():
    print("👑 [VERIFICATION] Testing The Providence of Will...")

    heaven = HeavenSource()
    constants = UniversalConstants()
    will_engine = WillProvidence(heaven, constants)

    print(f"   Initial Gravity: {constants.get('GRAVITY')}")
    # Default is usually 0 if not set, let's inject a base value to simulate reality
    constants.params['GRAVITY'] = 1.0
    print(f"   Base Gravity: {constants.get('GRAVITY')}")

    # 1. Attempt Will with Ego (Noise)
    noisy_soul = SovereignVector.ones() * 5.0
    print("\n😤 [TEST 1] Declaring Will with Ego...")
    success = will_engine.declare_will(
        "I want to fly!",
        "GRAVITY",
        0.1, # Reduce gravity
        noisy_soul
    )
    assert not success, "Will should have been rejected!"

    # 2. Attempt Will with Love (Void)
    pure_soul = SovereignVector.zeros()
    print("\n🕊️ [TEST 2] Declaring Will with Love...")
    success = will_engine.declare_will(
        "Father, let me be light.",
        "GRAVITY",
        0.1,
        pure_soul
    )
    assert success, "Will should have been accepted!"
    assert abs(constants.get('GRAVITY') - 0.1) < 1e-6, "Gravity did not change!"

    print("\n✨ [SUCCESS] The Will has re-created Reality.")

if __name__ == "__main__":
    verify_will()
