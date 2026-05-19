import unittest
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.Mind.resonance_engine import ResonanceEngine

class TestImagination(unittest.TestCase):
    def setUp(self):
        self.engine = ResonanceEngine()

    def test_prediction_logic(self):
        """Test that the engine correctly predicts outcomes."""
        # Set initial hunger
        self.engine.nodes["Hunger"].activation = 1.0
        
        # Predict outcome of Eat
        predicted = self.engine.predict_outcome("Eat")
        
        # Eat should reduce Hunger (-0.5)
        self.assertEqual(predicted["Hunger"], 0.5)
        
        # Predict outcome of Move
        predicted_move = self.engine.predict_outcome("Move")
        # Move doesn't affect Hunger in our simple model
        self.assertEqual(predicted_move["Hunger"], 1.0)

    def test_imagination_choice(self):
        """Test that imagination chooses the best future."""
        # Scenario: High Hunger
        inputs = {"Hunger": np.array([1.0, 1.0, 1.0])}
        
        # Reactive update would choose Eat because Hunger -> Eat connection is strong.
        # Imagination should ALSO choose Eat because it reduces Hunger (which is 'bad').
        
        action = self.engine.imagination_step(inputs)
        self.assertEqual(action, "Eat")
        
    def test_energy_conservation(self):
        """Test that imagination avoids waste."""
        # Scenario: No Hunger, Low Energy
        # Eat increases Energy (+0.2), Move decreases it (-0.1).
        # So Eat should be preferred even if not hungry, purely for energy gain?
        # Or maybe Rest?
        
        inputs = {"Energy": np.array([0.1, 0.1, 0.1])} # Low energy input? No, Energy node represents "Having Energy" usually.
        # Let's say inputs are empty (Resting state).
        inputs = {}
        
        # If we do nothing, state decays.
        # If we Eat, Energy +0.2.
        # If we Move, Energy -0.1.
        
        # Current logic: Score = -Hunger + Energy.
        # Eat: Hunger -0.5 (Good), Energy +0.2 (Good). Total gain.
        # Move: Energy -0.1 (Bad).
        
        action = self.engine.imagination_step(inputs)
        self.assertEqual(action, "Eat") # It realizes Eating is always good in this simple model!

if __name__ == '__main__':
    unittest.main()
