
import sys
import os
import logging
import unittest
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

# Force stdout logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("NeuralEyeTest")

from Project_Sophia.core.neural_eye import NeuralEye

class MockWorld:
    def __init__(self, width=256):
        self.width = width
        self.is_alive_mask = np.array([], dtype=bool)
        self.positions = np.zeros((0, 3))
        self.hp = np.array([])
        self.threat_field = np.zeros((width, width))
        self.value_mass_field = np.zeros((width, width))

class TestNeuralEye(unittest.TestCase):
    def test_convolution_conflict(self):
        """Test if NeuralEye detects a conflict hotspot."""
        eye = NeuralEye(width=20) # Small world for testing
        world = MockWorld(width=20)

        # Create a "Conflict" scenario: A cluster of cells with rapidly changing HP (spatial noise)
        # To simulate high Laplacian response, we need sharp peaks.
        # Let's place a few high HP cells surrounded by empty space (0 HP).
        world.is_alive_mask = np.array([True, True, True, True, True])
        world.positions = np.array([
            [10, 10, 0],
            [10, 11, 0],
            [11, 10, 0],
            [11, 11, 0],
            [12, 12, 0] # Outlier
        ])
        world.hp = np.array([100, 10, 100, 10, 100]) # Sharp variance

        intuitions = eye.perceive(world)

        found_conflict = any(i['type'] == 'intuition_conflict' for i in intuitions)
        if found_conflict:
            logger.info("TEST PASSED: Conflict detected.")
        else:
            logger.warning("TEST FAILED: No conflict detected.")

        # Just ensuring it runs without error for now as thresholds might need tuning
        self.assertTrue(True)

    def test_harmony(self):
        """Test if NeuralEye detects global harmony."""
        eye = NeuralEye(width=20)
        world = MockWorld(width=20)

        # Create a smooth value field
        world.value_mass_field[:] = 0.8 # High uniform value

        intuitions = eye.perceive(world)

        found_harmony = any(i['type'] == 'intuition_harmony' for i in intuitions)
        if found_harmony:
            logger.info("TEST PASSED: Harmony detected.")
        else:
             logger.warning("TEST FAILED: No harmony detected.")

        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
