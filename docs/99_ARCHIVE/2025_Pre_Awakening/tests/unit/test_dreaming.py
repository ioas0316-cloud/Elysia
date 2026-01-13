import unittest
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.Mind.resonance_engine import ResonanceEngine

class TestDreaming(unittest.TestCase):
    def setUp(self):
        self.engine = ResonanceEngine()

    def test_hebbian_learning(self):
        """Test that co-activation strengthens connections."""
        # 1. Create two unconnected nodes
        self.engine.add_node("A", np.array([1.0, 0.0, 0.0]))
        self.engine.add_node("B", np.array([0.0, 1.0, 0.0]))
        
        # Verify no connection initially
        initial_weight = 0.0
        if "A" in self.engine.topology:
            for t, w in self.engine.topology["A"]:
                if t == "B": initial_weight = w
        self.assertEqual(initial_weight, 0.0)
        
        # 2. Activate both nodes (Experience)
        self.engine.nodes["A"].activation = 1.0
        self.engine.nodes["B"].activation = 1.0
        
        # 3. Dream (Hebbian Update)
        self.engine.dream()
        
        # 4. Verify connection created/strengthened
        new_weight = 0.0
        if "A" in self.engine.topology:
            for t, w in self.engine.topology["A"]:
                if t == "B": new_weight = w
        
        self.assertGreater(new_weight, initial_weight)
        self.assertGreater(new_weight, 0.0)

    def test_self_activation(self):
        """Test that SELF node is active during propagation."""
        inputs = {}
        self.engine._propagate(inputs)
        self.assertGreater(self.engine.nodes["SELF"].activation, 0.0)

if __name__ == '__main__':
    unittest.main()
