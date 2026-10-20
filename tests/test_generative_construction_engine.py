"""
Unit and Integration Tests for Generative Construction Engine.
"""

import unittest
import numpy as np

from core.physics.generative_construction_engine import (
    GenerativeConstructionEngine,
    PrismLens,
    ScaleFreePhaseFocalTuner,
    ConstructionState,
    PrismRefractionResult
)


class TestGenerativeConstructionEngine(unittest.TestCase):

    def setUp(self):
        self.engine = GenerativeConstructionEngine(dimension=3)
        self.prism_lens = PrismLens(dimension=3)
        self.focal_tuner = ScaleFreePhaseFocalTuner(num_scales=4)

    def test_prism_lens_refraction_and_label_less_differentiation(self):
        """Test that PrismLens refracts unlabelled signals into chromatic signatures and structural identities."""
        # 1. High frequency torque signal
        torque_wave = np.sin(np.linspace(0, 100, 256)) + 2.0 * np.cos(np.linspace(0, 300, 256))
        refraction_torque = self.prism_lens.refract(torque_wave)

        self.assertIsInstance(refraction_torque, PrismRefractionResult)
        self.assertEqual(len(refraction_torque.chromatic_vector), 3)
        self.assertGreater(refraction_torque.phase_torque_response, 0.0)

        # 2. Harmonic smooth causal signal
        harmonic_wave = np.sin(np.linspace(0, 2 * np.pi, 256))
        refraction_harmonic = self.prism_lens.refract(harmonic_wave)

        self.assertIsInstance(refraction_harmonic, PrismRefractionResult)
        self.assertGreater(refraction_harmonic.chromatic_vector[1], 0.3)  # High order (Blue)

    def test_scale_free_phase_focal_tuner(self):
        """Test lossless scale tuning and phase focal precision across micro-macro scales."""
        micro_wave = np.sin(np.linspace(0, 10 * np.pi, 128))
        target_scale = 100.0

        amplified_signal, focal_precision = self.focal_tuner.tune_focal_point(micro_wave, target_scale)

        self.assertEqual(len(amplified_signal), len(micro_wave))
        self.assertGreater(np.max(np.abs(amplified_signal)), np.max(np.abs(micro_wave)))
        self.assertGreaterEqual(focal_precision, 0.0)

    def test_construct_step_and_constraint_relaxation(self):
        """Test generative construction step and phase-locking relaxation."""
        state0 = self.engine.construct_step()
        self.assertEqual(state0.step, 0)
        self.assertEqual(state0.positions.shape, (8, 3))

        state1 = self.engine.construct_step(external_torque_input=np.ones(64))
        self.assertEqual(state1.step, 1)
        self.assertEqual(len(self.engine.trajectory_history), 2)

    def test_eye_of_agamotto_trajectory_scrubbing(self):
        """Test O(1) Eye of Agamotto trajectory scrubbing and exact state restoration."""
        # Construct 5 steps
        for i in range(5):
            self.engine.construct_step()

        self.assertEqual(len(self.engine.trajectory_history), 5)
        initial_pos_step_1 = self.engine.trajectory_history[1].positions.copy()

        # Scrub back to step 1
        scrubbed_state = self.engine.scrub_trajectory_eye_of_agamotto(target_step=1)

        self.assertEqual(scrubbed_state.step, 1)
        np.testing.assert_allclose(self.engine.boundary_positions, initial_pos_step_1, atol=1e-12)

    def test_integrate_embodied_tool(self):
        """Test physical tool torque integration into internal sensory manifold."""
        tool_signal = np.sin(np.linspace(0, 50, 128)) * 3.0
        result = self.engine.integrate_embodied_tool(tool_signal)

        self.assertIn("refraction", result)
        self.assertGreater(result["phase_torque_magnitude"], 0.0)
        self.assertEqual(result["embodied_status"], "Tool integrated as internal sensory manifold")

    def test_align_symbolic_wave_pre_linguistic_manifold(self):
        """Test symbolic wave mapping over pre-linguistic causal manifold."""
        token = "RotationalTorque"
        waveform = np.sin(np.linspace(0, 20, 128))
        result = self.engine.align_symbolic_wave(token, waveform)

        self.assertEqual(result["token"], token)
        self.assertIn(token, self.engine.symbolic_anchor_map)
        self.assertEqual(len(self.engine.symbolic_anchor_map[token]), 3)


if __name__ == "__main__":
    unittest.main()
