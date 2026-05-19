import unittest
import torch
import numpy as np
from Core.Cognition.prism import DoubleHelixPrism, DoubleHelixWave, SevenChannelQualia

class TestDoubleHelixPrism(unittest.TestCase):
    def setUp(self):
        self.prism = DoubleHelixPrism()

    def test_refract_weight_structure(self):
        """
        Verify that a refracted weight returns a valid DoubleHelixWave
        with Pattern and Principle strands.
        """
        print("\n🧪 [Test] Prism Refraction (Weight -> Wave)")

        # Create a mock weight (random signal)
        mock_weight = torch.randn(1024)

        # Refract
        wave = self.prism.refract_weight(mock_weight, "layer.test")

        # Check Types
        self.assertIsInstance(wave, DoubleHelixWave)
        self.assertIsInstance(wave.pattern_strand, torch.Tensor)
        self.assertIsInstance(wave.principle_strand, torch.Tensor)

        # Check Dimensions
        # Principle should be 7-Dimensional (The 7 Qualia)
        self.assertEqual(wave.principle_strand.shape[0], 7)

        print(f"   ✅ Pattern Shape: {wave.pattern_strand.shape}")
        print(f"   ✅ Principle Shape: {wave.principle_strand.shape} (7-Channel)")
        print(f"   ✅ Phase: {wave.phase:.4f} rad")

    def test_qualia_spectrum(self):
        """
        Verify that different signals produce different Qualia signatures.
        (Low freq signal vs High freq signal)
        """
        print("\n🧪 [Test] Qualia Spectrum Sensitivity")

        # 1. Low Frequency Signal (Sine wave)
        t = torch.linspace(0, 10, 1024)
        low_freq = torch.sin(t) # 1 oscillation

        # 2. High Frequency Signal (Noise/Rapid Sine)
        high_freq = torch.sin(t * 100) # 100 oscillations

        wave_low = self.prism.refract_weight(low_freq, "low")
        wave_high = self.prism.refract_weight(high_freq, "high")

        q_low = wave_low.principle_strand
        q_high = wave_high.principle_strand

        print(f"   🌊 Low Freq Qualia: {q_low.tolist()}")
        print(f"   🌊 High Freq Qualia: {q_high.tolist()}")

        # Causal (Index 0) corresponds to Low Freq bands in our mapping
        # So Low Freq signal should have high Causal score.
        # But wait, our mapping in Prism is:
        # bands[0] -> Causal.
        # So yes, Low Freq signal -> Energy in bands[0] -> High Causal.

        # Let's verify that they are DIFFERENT
        dist = torch.norm(q_low - q_high).item()
        print(f"   📏 Spectral Distance: {dist:.4f}")
        self.assertGreater(dist, 0.1, "Prism failed to distinguish Low vs High frequency signals.")

if __name__ == '__main__':
    unittest.main()
