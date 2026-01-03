"""
Verify Omni-Voxel: The Wireless Memory Test
-------------------------------------------
Demonstrates that data can be accessed via "Resonance" (Wireless)
rather than "Address" (Wired).
"""

import sys
import os

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.Foundation.Memory.Orb.orb_manager import OrbManager
from Core.Foundation.Protocols.pulse_protocol import WavePacket, PulseType

def test_wireless_recall():
    print("🧪 [Test] Starting Omni-Voxel Verification...")

    manager = OrbManager()

    # 1. Create Concepts (The Voxels)
    # Apple = Red (400Hz)
    # Sky = Blue (600Hz)
    # Fire = Red (410Hz) - Close to Apple
    apple = manager.create_orb("Apple", frequency=400.0)
    sky = manager.create_orb("Sky", frequency=600.0)
    fire = manager.create_orb("Fire", frequency=410.0)

    print(f"   Created Orbs: {apple}, {sky}, {fire}")

    # 2. Broadcast "Red" Signal (Wireless Search)
    # We are NOT asking for "Apple" by name. We are broadcasting "400Hz".
    print("\n📡 [Broadcast] Sending 'Red Frequency' (400Hz)...")

    red_pulse = WavePacket(
        sender="Tester",
        type=PulseType.INTENTION_SHIFT,
        payload={"frequency": 400.0}
    )

    results = manager.broadcast(red_pulse)

    # 3. Verify Resonance
    print(f"   Resonated Orbs: {[orb.name for orb in results]}")

    # Expectation: Apple (Exact match) and Fire (Close match) should resonate.
    # Sky (Far match) should not.

    names = [orb.name for orb in results]

    if "Apple" in names and "Fire" in names:
        print("✅ [Success] 'Red' pulse triggered both Apple and Fire!")
    else:
        print("❌ [Failure] Resonance logic failed.")
        exit(1)

    if "Sky" not in names:
        print("✅ [Success] 'Blue' Sky did not resonate with Red pulse.")
    else:
        print("❌ [Failure] Sky resonated incorrectly.")
        exit(1)

    print("\n✨ [Conclusion] Wireless/Content-Addressable Memory Verified.")

if __name__ == "__main__":
    test_wireless_recall()
