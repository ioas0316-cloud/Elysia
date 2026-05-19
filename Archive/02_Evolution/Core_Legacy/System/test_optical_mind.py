"""
Test: Optical Mind
==================
Verifies Module A: Prism Engine (Fractal Optics).
"""

import sys
import os
import math
import unittest

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.Phenomena.fractal_optics import PrismEngine, WavePacket

class TestOpticalMind(unittest.TestCase):

    def setUp(self):
        self.prism = PrismEngine()

    def test_vectorization(self):
        """Test if text is converted to a valid WavePacket."""
        text = "Elysia"
        wave = self.prism.vectorize(text)

        print(f"\n--- Vectorization: '{text}' ---")
        print(f"Vector Norm: {wave.intensity():.4f}")
        print(f"Phase: {wave.phase:.4f} rad")

        self.assertAlmostEqual(wave.intensity(), 1.0, places=2)

    def test_interference(self):
        """Test if changing the Rotor Angle changes the resonance outcome."""
        text = "Truth"
        wave = self.prism.vectorize(text)

        # 1. Angle 0
        results_0 = self.prism.traverse(wave, incident_angle=0.0)
        best_path_0 = results_0[0][0]

        # 2. Angle PI (Opposite Perspective)
        results_pi = self.prism.traverse(wave, incident_angle=math.pi)
        best_path_pi = results_pi[0][0]

        print(f"\n--- Interference Test: '{text}' ---")
        print(f"Angle 0.0 best path: {best_path_0} (Score: {results_0[0][1]:.2f})")
        print(f"Angle PI  best path: {best_path_pi} (Score: {results_pi[0][1]:.2f})")

        # The paths should be different because the constructive interference happens elsewhere
        self.assertNotEqual(best_path_0, best_path_pi)

    def test_fractal_depth(self):
        """Test if the traversal goes deep enough."""
        wave = self.prism.vectorize("Depth")
        results = self.prism.traverse(wave, 0.0)
        path = results[0][0]

        # We expect a path of length 3 (e.g., "0->1->2")
        depth = len(path.split("->"))
        self.assertEqual(depth, self.prism.max_depth)

if __name__ == '__main__':
    unittest.main()
