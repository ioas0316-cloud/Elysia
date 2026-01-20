import unittest
import torch
import numpy as np
from Core.L7_Spirit.Monad.monad_core import Monad, FractalRule

class TestMonadSovereignty(unittest.TestCase):
    def setUp(self):
        # Create a Monad with a specific "Order/Structure" intent
        # 7D: [Physical, Functional, Phenomenal, Causal, Mental, Structural, Spiritual]
        # High Structural (idx 5) and Mental (idx 4) intent
        order_intent = [0.1, 0.2, 0.1, 0.1, 0.8, 0.9, 0.5]
        self.monad = Monad(seed="OrderMonad", intent_vector=order_intent)

    def test_resonance_acceptance(self):
        """
        Test Case: Feed 'Structure' input -> Should Absorb/Resonate.
        """
        print("\n🧪 [Test] Monad Resonance (Order vs Order)")

        # Input: High Structural signal
        input_signal = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.7, 0.95, 0.4], dtype=torch.float32)

        accepted, score = self.monad.resonate(input_signal)

        print(f"   🌊 Input: Structural Wave")
        print(f"   ✨ Resonance Score: {score:.4f}")

        self.assertTrue(accepted, "Monad should accept resonant input.")
        self.assertGreater(score, 0.8, "Resonance score should be high for similar vectors.")

    def test_resonance_rejection(self):
        """
        Test Case: Feed 'Chaos' (Opposite) input -> Should Reject.
        """
        print("\n🧪 [Test] Monad Rejection (Order vs Chaos)")

        # Input: High Physical/Phenomenal (Sensation/Chaos), Low Structural
        # Vector almost orthogonal or opposite to Order
        input_signal = torch.tensor([0.9, 0.8, 0.9, 0.1, 0.1, 0.0, 0.1], dtype=torch.float32)

        accepted, score = self.monad.resonate(input_signal)

        print(f"   🌊 Input: Chaos/Sensation Wave")
        print(f"   🛡️ Resonance Score: {score:.4f}")

        self.assertFalse(accepted, "Monad should reject dissonant input.")
        self.assertLess(score, 0.6, "Resonance score should be low for dissimilar vectors.")

if __name__ == '__main__':
    unittest.main()
