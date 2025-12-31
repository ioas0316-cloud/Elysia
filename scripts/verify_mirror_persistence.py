
import sys
import os
import unittest
import json
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from Core.Cognition.Topology.mirror_surface import MirrorSurface
from Core.Cognition.Wisdom.wisdom_store import WisdomStore

class MockWisdomStore(WisdomStore):
    def __init__(self):
        self.values = {"Love": 0.9}

    def get_decision_weight(self, key: str) -> float:
        return self.values.get(key, 0.5)

class TestMirrorPersistence(unittest.TestCase):
    def setUp(self):
        self.wisdom = MockWisdomStore()
        self.test_memory_path = "data/test_mirror_memory.json"
        # Ensure clean state
        if os.path.exists(self.test_memory_path):
            os.remove(self.test_memory_path)

    def tearDown(self):
        if os.path.exists(self.test_memory_path):
            os.remove(self.test_memory_path)

    def test_memory_persistence(self):
        """Test if the mirror remembers its history across instances."""

        # 1. Create First Mirror and Reflect
        mirror1 = MirrorSurface(self.wisdom, memory_path=self.test_memory_path)
        mirror1.reflect("I am learning.")
        mirror1.reflect("The world is strange.")

        # Check in-memory state
        self.assertEqual(len(mirror1.history), 2)
        self.assertGreater(mirror1.patina_factor, 0.0)
        original_patina = mirror1.patina_factor

        print(f"\n[Persistence] Mirror 1 Patina: {original_patina}")

        # 2. Create Second Mirror (simulating restart) pointing to same file
        mirror2 = MirrorSurface(self.wisdom, memory_path=self.test_memory_path)

        # Check if it loaded the state
        self.assertEqual(len(mirror2.history), 2)
        self.assertEqual(mirror2.patina_factor, original_patina)
        print(f"[Persistence] Mirror 2 Loaded Patina: {mirror2.patina_factor}")

        # 3. Add more to Mirror 2
        mirror2.reflect("I remember you.")
        self.assertEqual(len(mirror2.history), 3)
        self.assertGreater(mirror2.patina_factor, original_patina)

if __name__ == '__main__':
    unittest.main()
