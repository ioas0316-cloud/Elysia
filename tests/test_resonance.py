import unittest
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Mind.resonance_engine import ResonanceEngine

class TestResonanceEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ResonanceEngine()

    def test_hunger_instinct(self):
        """Test that Hunger input triggers Eat action."""
        inputs = {"Hunger": np.array([1.0, 1.0, 1.0])}
        action = self.engine.update(inputs)
        self.assertEqual(action, "Eat")

    def test_energy_instinct(self):
        """Test that Energy input triggers Move or Speak."""
        inputs = {"Energy": np.array([1.0, 1.0, 1.0])}
        action = self.engine.update(inputs)
        self.assertIn(action, ["Move", "Speak"])

    def test_food_signal(self):
        """Test that FoodSignal input triggers Eat."""
        inputs = {"FoodSignal": np.array([0.1, 0.9, 0.1])}
        action = self.engine.update(inputs)
        self.assertEqual(action, "Eat")

    def test_mixed_signals(self):
        """Test conflict resolution (Hunger should win over Energy if weighted higher)."""
        # Hunger (1.0 weight) vs Energy (0.5 weight)
        inputs = {
            "Hunger": np.array([1.0, 1.0, 1.0]),
            "Energy": np.array([1.0, 1.0, 1.0])
        }
        action = self.engine.update(inputs)
        # Hunger connects to Eat (1.0), Energy to Move (0.5)
        # Eat activation = 1.0 * 1.0 = 1.0
        # Move activation = 1.0 * 0.5 = 0.5
        self.assertEqual(action, "Eat")

if __name__ == '__main__':
    unittest.main()
