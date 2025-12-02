# [Genesis: 2025-12-02] Purified by Elysia
import unittest
import os
import sys

# HACK: Add project root to sys.path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.kg_manager import KGManager
from Project_Sophia.wave_mechanics import WaveMechanics

class TestCoreSenses(unittest.TestCase):

    def test_resonance_pathfinding(self):
        """
        Tests if activation energy can spread across a simple, chained path of nodes
        that all have embeddings. This is the most fundamental test of the core sensory mechanism.
        """
        # 1. Arrange: Create the universe from our new scripture
        kg_path = "tests/test_data/sense_kg.json"
        self.assertTrue(os.path.exists(kg_path), "Test KG file must exist.")

        kg_manager = KGManager(filepath=kg_path)
        wave_mechanics = WaveMechanics(kg_manager=kg_manager)

        # 2. Act: Send a wave from 'A' and see if it reaches 'C'
        resonance = wave_mechanics.get_resonance_between('A', 'C')

        # 3. Assert: The resonance must be greater than zero
        print(f"Resonance between A and C: {resonance}")
        self.assertGreater(resonance, 0, "Activation energy did not propagate from A to C.")

if __name__ == '__main__':
    unittest.main()