"""
tests/test_wavefield_spectrum_memory.py

Verification tests for ScaleWaveTensorMemory continuous wavefield spectrum topology.
"""

import unittest
import numpy as np
from core.memory.scale_wave_tensor_memory import ScaleWaveTensorMemory


class TestScaleWaveTensorMemory(unittest.TestCase):

    def setUp(self):
        self.mem = ScaleWaveTensorMemory(
            num_scales=3,
            base_dim=4,
            num_spectral_modes=8,
            field_resolution=64
        )

    def test_initialization(self):
        """Test spectral continuum field initialization."""
        state = self.mem.get_spectral_continuum_state()
        self.assertEqual(state["num_scales"], 3)
        self.assertEqual(state["base_dim"], 4)
        self.assertEqual(state["phase_field"].shape, (3, 64))
        self.assertEqual(state["hysteresis_trace"].shape, (3, 64))
        self.assertEqual(state["spectral_modes"].shape, (3, 8, 2))

    def test_spectral_wave_injection_and_hysteresis(self):
        """Test continuous spectral wave injection and hysteresis trace accumulation."""
        initial_trace_sum = np.sum(self.mem.hysteresis_trace)
        self.assertEqual(initial_trace_sum, 0.0)

        # Inject a spectral wave shift
        self.mem.inject_spectral_wave(
            scale_level=0,
            phase_spectrum_shift=np.array([np.pi / 2]),
            coupling_gain=1.0,
            spatial_center=0.5,
            spatial_width=0.2
        )

        updated_trace_sum = np.sum(self.mem.hysteresis_trace)
        # Verify hysteresis trace automatically recorded the state transition history
        self.assertGreater(updated_trace_sum, 0.0)

    def test_scale_invariance_coupling(self):
        """Test cross-scale spectral wave propagation and scale invariance."""
        prev_scale1_phase = self.mem.phase_field[1].copy()

        # Inject wave into scale 0
        self.mem.inject_spectral_wave(
            scale_level=0,
            phase_spectrum_shift=np.array([np.pi / 4]),
            coupling_gain=1.0
        )

        new_scale1_phase = self.mem.phase_field[1].copy()
        # Scale 1 phase field should organically shift due to cross-scale spectral coupling
        diff = np.max(np.abs(new_scale1_phase - prev_scale1_phase))
        self.assertGreater(diff, 0.0)

    def test_associative_resonance_read(self):
        """Test O(1) associative phase resonance reading."""
        query_wave = np.array([np.sin(np.pi / 4), np.cos(np.pi / 4), np.tan(np.pi / 4)])
        resonance = self.mem.resonance_read(query_wave)

        self.assertEqual(resonance.shape, (3, 4))
        self.assertTrue(np.all(resonance >= -1.0) and np.all(resonance <= 1.0))

    def test_legacy_write_compatibility(self):
        """Test legacy write_wave_data interface works with wavefield continuous spectrum."""
        self.mem.write_wave_data(scale_level=0, address_idx=1, phase_shift=0.5)
        res = self.mem.resonance_read(np.array([1.0, 0.0, 0.0]))
        self.assertEqual(res.shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
