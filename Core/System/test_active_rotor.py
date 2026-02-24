"""
Test: Active Rotor
==================
Verifies Module B: Active Rotor (Cognitive Tuning).
"""

import sys
import os
import math
import unittest

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.System.active_rotor import ActiveRotor
from Core.Phenomena.fractal_optics import PrismEngine

class TestActiveRotor(unittest.TestCase):

    def setUp(self):
        self.rotor = ActiveRotor()
        self.prism = PrismEngine()

    def test_tuning(self):
        """Test if the rotor finds a resonant angle."""
        text = "Insight"
        wave = self.prism.vectorize(text)

        print(f"\n--- Tuning for: '{text}' ---")
        best_angle, score, path = self.rotor.tune(wave, self.prism)

        print(f"Best Angle: {best_angle:.2f} rad")
        print(f"Resonance Score: {score:.2f}")
        print(f"Path: {path}")

        # It should find something better than random chance (usually around 1.0)
        # Interference boosts it above 1.0
        # With 0.98 decay, scores should be consistently higher
        self.assertTrue(score > 1.0)
        self.assertTrue(len(path) > 0)

    def test_lock_on(self):
        """Test if the rotor state updates after tuning."""
        text = "Focus"
        wave = self.prism.vectorize(text)

        angle, _, _ = self.rotor.tune(wave, self.prism)

        # Check if the rotor physically moved to that angle (in degrees)
        expected_deg = (angle * 180.0 / math.pi) % 360.0
        print(f"\n--- Lock On Test ---")
        print(f"Tuned Angle (rad): {angle:.2f}")
        print(f"Rotor Angle (deg): {self.rotor.current_angle:.2f}")

        self.assertAlmostEqual(self.rotor.current_angle, expected_deg, places=1)
        # RPM should boost on success
        self.assertEqual(self.rotor.target_rpm, 60.0)

if __name__ == '__main__':
    unittest.main()
