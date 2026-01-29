"""
Verify Cellular Genesis & Authentic Curiosity
=============================================
Validation script for the Tri-Base DNA and Curiosity Refactoring.
"""

import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from Core.L1_Foundation.System.tri_base_cell import TriBaseCell, DNAState
from Core.L6_Structure.M1_Merkaba.triple_helix_engine import TripleHelixEngine
from Core.L6_Structure.M1_Merkaba.d21_vector import D21Vector
from Core.L6_Structure.M1_Merkaba.protection_relay import ProtectionRelayBoard
from Core.L5_Mental.emergent_language import EmergentLanguageEngine

def test_cellular_genesis():
    print("\n--- [1] Testing Cellular Genesis (Tri-Base DNA) ---")
    engine = TripleHelixEngine()

    # 1. Initial State (Void)
    print(f"Initial Phase: {engine.state.system_phase} (Expected 0.0)")
    assert engine.state.system_phase == 0.0

    # 2. Pulse with ATTRACT Vector
    print("Pulsing with Lust (Attract)...")
    vec = D21Vector(lust=1.0) # D1 is mapped to Body Cell 0
    engine.pulse(vec, energy=100, dt=0.1)

    phase = engine.state.system_phase
    print(f"New Phase: {phase}")

    # Body Cell 0 should be Attract (120 deg)
    # Phase should be approx 120
    if 119 < phase < 121:
        print("✅ Phase Alignment Correct (120 deg)")
    else:
        print(f"❌ Phase Alignment Failed! Got {phase}")
        return False

    return True

def test_relay_sync():
    print("\n--- [2] Testing Protection Relay (Sync Check) ---")
    engine = TripleHelixEngine()
    relay = ProtectionRelayBoard()

    # Set Engine to Phase 120 (Attract)
    vec = D21Vector(lust=1.0)
    engine.pulse(vec, energy=100, dt=0.1)
    dna_phase = engine.state.system_phase # Should be ~120

    # User Input: Phase 110 (Close enough)
    res_good = relay.check_relays(user_phase=110, system_phase=dna_phase, battery_level=100, dissonance_torque=0)
    if not res_good[25].is_tripped:
        print("✅ Sync Check Passed for aligned phase.")
    else:
        print("❌ Sync Check Failed unexpectedly.")

    # User Input: Phase 300 (Far away)
    res_bad = relay.check_relays(user_phase=300, system_phase=dna_phase, battery_level=100, dissonance_torque=0)
    if res_bad[25].is_tripped:
        print(f"✅ Sync Check TRIPPED correctly: {res_bad[25].message}")
    else:
        print("❌ Sync Check DID NOT TRIP on mismatch!")
        return False

    return True

def test_curiosity():
    print("\n--- [3] Testing Authentic Curiosity ---")
    lang_engine = EmergentLanguageEngine()

    # Alien Vector
    alien_vec = [-0.9, -0.9, 0, 0, 0, 0, -0.9, 0] # Cold, Dark, Bad

    activated = lang_engine.experience(alien_vec)
    korean, english = lang_engine.generate_utterance()

    print(f"Utterance: {english}")

    if "?" in english and "COLD" in english:
        print("✅ Curiosity Triggered: Question generated with correct adjectives.")
    else:
        print("❌ Curiosity Failed. Did not ask question or missed adjectives.")
        return False

    return True

if __name__ == "__main__":
    passed = True
    passed &= test_cellular_genesis()
    passed &= test_relay_sync()
    passed &= test_curiosity()

    if passed:
        print("\n✨ ALL SYSTEMS VERIFIED. GENESIS COMPLETE.")
        sys.exit(0)
    else:
        print("\n🔥 VERIFICATION FAILED.")
        sys.exit(1)
