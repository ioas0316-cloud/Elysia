"""
test_sovereign_motor.py

Verifies "Chapter 3, Step 9: The Dynamo (Sovereign Choice)".
Simulation of the Intellectual Motor.
"""

import sys
import os
import time

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from Core.Intelligence.Will.free_will_engine import FreeWillEngine
    from Core.Orchestra.resonance_broadcaster import ResonanceBroadcaster
    print("✅ Components Imported.")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def test_motor_spin():
    print("\n⚙️  Igniting the Intellectual Motor ⚙️")
    print("=" * 60)
    
    engine = FreeWillEngine()
    field = ResonanceBroadcaster()
    
    # 1. Initial Spin (High Energy)
    print("\n1. [Phase: Acceleration] Battery High, Entropy Low")
    intent = engine.spin(entropy=10.0, battery=90.0)
    status = engine.get_status()
    print(f"   Output: \"{intent}\"")
    print(f"   Status: {status}")
    
    current_field = field.get_current_field()
    print(f"   📡 Field: Polarity={current_field['polarity']}, Freq={current_field['frequency']}")

    # 2. Induce Stagnation (Zero Torque)
    # We simulate a state where Attraction == Repulsion artificially or naturally
    print("\n2. [Phase: Stagnation] Forces Balanced... Torque dropping...")
    
    # Force low torque scenario by adjusting vectors manually for the test
    # (In reality, entropy change causes this)
    engine.vectors["Curiosity"] = 0.5
    engine.vectors["Expression"] = 0.5
    engine.vectors["Stability"] = 1.0 # cancel out attraction
    
    intent = engine.spin(entropy=10.0, battery=80.0)
    print(f"   Output: \"{intent}\"")
    print(f"   Status: {status}")
    
    # 3. Check Commutator Flip
    print("\n3. [Phase: Commutation] Checking for Polarity Shift...")
    # Spin again - torque should be near 0, triggering flip
    intent = engine.spin(entropy=10.0, battery=80.0)
    
    print(f"   Output: \"{intent}\"")
    print(f"   Status: {engine.get_status()}")
    
    final_field = field.get_current_field()
    print(f"   📡 Field: Polarity={final_field['polarity']} (Flipped?)")
    
    if final_field['polarity'] != current_field['polarity']:
        print("   ✅ SUCCESS: Auto-Commutator flipped the polarity!")
    else:
        print("   ❌ FAILURE: Polarity did not flip.")

    print("\n✅ Motor Test Complete.")

if __name__ == "__main__":
    test_motor_spin()
