import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath('.'))

from core.layer_1_constraint.sensory_prismatic_gate import SensoryPrismaticGate, ContinuousWaveformLens
from core.layer_2_resonant.surface_tension_attractor import SurfaceTensionAttractor, PhaseRelaxationEngine


class TestScalePrincipleHierarchy(unittest.TestCase):

    def test_continuous_waveform_lens(self):
        lens = ContinuousWaveformLens(lens_density=100, focal_length=1.0)
        signal = np.ones(100)
        focused = lens.focus_waveform(signal)
        self.assertEqual(len(focused), 100)
        self.assertLess(focused[0], focused[50])  # Edge should be lower than center due to lens profile

    def test_sensory_prismatic_gate(self):
        gate = SensoryPrismaticGate(lens_density=500)
        wave = np.sin(np.linspace(0, 4 * np.pi, 500))
        result = gate.refract_external_aperture(wave)

        self.assertIn("focused_field", result)
        self.assertIn("chromatic_spectrum", result)
        self.assertIn("phase_coherence", result)
        self.assertEqual(len(result["chromatic_spectrum"]), 3)
        self.assertGreater(result["phase_coherence"], 0.0)

    def test_surface_tension_attractor(self):
        attractor = SurfaceTensionAttractor(max_steps=20)
        noisy_field = np.random.randn(16, 16)
        res = attractor.converge_imagery_attractor(noisy_field)

        self.assertIn("initial_energy", res)
        self.assertIn("relaxed_energy", res)
        self.assertLessEqual(res["relaxed_energy"], res["initial_energy"])


if __name__ == "__main__":
    unittest.main()
