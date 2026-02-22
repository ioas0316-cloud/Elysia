import unittest
import numpy as np
from Core.S1_Body.L6_Structure.M6_Architecture.holographic_memory import HolographicMemory
from Core.S1_Body.L6_Structure.M6_Architecture.holographic_persistence import HolographicPersistence

import shutil
from pathlib import Path

class TestPersistence(unittest.TestCase):

    def setUp(self):
        # Use a temporary directory for tests
        self.test_dir = "data/test_persistence_tmp"
        p = HolographicPersistence(storage_path=self.test_dir)
        p.clear()

    def tearDown(self):
        # Clean up after test
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)

    def test_reincarnation(self):
        """
        Verify that knowledge survives the death of the instance.
        """
        # 1. Life 1: Learning
        # Note: HolographicMemory needs to support passing storage_path down to persistence
        # We manually patch it for the test if __init__ doesn't support it yet,
        # or we update HolographicMemory to accept it.
        # Ideally, we update HolographicMemory.

        # Monkey-patching for now to avoid changing signature if not needed,
        # but let's assume we pass it or set it.

        brain1 = HolographicMemory(dimension=64)
        brain1.persistence = HolographicPersistence(storage_path=self.test_dir) # Inject

        brain1.imprint("Apple", intensity=1.0, quality="RED")
        brain1.save_state() # Freeze

        # 2. Death
        del brain1

        # 3. Life 2: Rebirth
        brain2 = HolographicMemory(dimension=64)
        brain2.persistence = HolographicPersistence(storage_path=self.test_dir) # Inject
        brain2._thaw_memory() # Manually trigger thaw since __init__ used default

        # Check if memory persists
        # We broadcast RED to see if Apple responds
        brain2.frequency_map["RED"] = 0.0 # Force phase for test stability

        # Note: In a real reload, frequency_map is also loaded.
        # But since we use hash(), it might drift if python restarts.
        # However, pickle saves the exact map, so it should be fine within same session.

        # Check direct resonance
        (concept, amp, phase) = brain2.resonate("Apple")

        # If persistence worked, amplitude should be high
        # Note: 'resonate' checks against NEUTRAL quality.
        # Imprint was with RED quality.
        # This results in a Phase Shift, but Amplitude should be preserved (1.0).

        print(f"\n[PERSISTENCE] Resurrected 'Apple' Amplitude: {amp:.4f}")
        print(f"[PERSISTENCE] Manifold Energy: {np.sum(np.abs(brain2.manifold)):.2f}")

        self.assertGreater(amp, 0.9)

if __name__ == '__main__':
    unittest.main()
