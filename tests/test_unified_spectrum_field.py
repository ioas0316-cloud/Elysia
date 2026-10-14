"""
Unit tests for UnifiedSpectrumFieldEngine in core/physics/unified_spectrum_field.py
"""

import unittest
import numpy as np
from core.physics.unified_spectrum_field import (
    UnifiedSpectrumFieldEngine,
    PhysicalSpectrumState,
    MacroRotor,
    RetrocausalFieldResult
)


class TestUnifiedSpectrumFieldEngine(unittest.TestCase):

    def setUp(self):
        self.engine = UnifiedSpectrumFieldEngine(resolution=128)
        t = np.linspace(0, 2 * np.pi, 128)
        self.test_wave = np.sin(3 * t) + 0.5 * np.cos(7 * t)
        self.engine.set_field_state(self.test_wave)

    def test_physical_quantity_spectrum_grounding(self):
        """Verify physical quantity extraction from continuous wave spectrum."""
        phys_state = self.engine.ground_physical_quantities()
        self.assertIsInstance(phys_state, PhysicalSpectrumState)
        self.assertGreater(phys_state.energy_amplitude_power, 0.0)
        self.assertGreaterEqual(phys_state.distance_phase_shift, 0.0)
        self.assertGreaterEqual(phys_state.velocity_freq_rate, 0.0)
        self.assertGreaterEqual(phys_state.force_interference_pressure, 0.0)
        self.assertGreaterEqual(phys_state.temperature_dispersion, 0.0)

    def test_formula_spectrum_transformations(self):
        """Verify dynamic formula transformations on the wave field stack."""
        initial_state = self.engine.field_state.copy()

        # 1. Euler Rotation
        rot_state = self.engine.apply_euler_rotation(np.pi / 2)
        self.assertFalse(np.allclose(initial_state, rot_state))

        # 2. Log Scale Compression
        log_state = self.engine.apply_log_scale_compression(base=2.0)
        self.assertEqual(len(log_state), 128)

        # 3. Force Acceleration (F = ma)
        acc_state = self.engine.apply_force_acceleration(mass=1.0, force_amplitude=5.0)
        self.assertEqual(len(acc_state), 128)

        # 4. Lagrangian Collapse
        potential = np.ones(128) * 0.5
        collapse_state, action_val = self.engine.apply_lagrangian_wave_collapse(potential)
        self.assertEqual(len(collapse_state), 128)
        self.assertIsInstance(action_val, float)

        # 5. Thermal Equilibrium
        eq_state = self.engine.apply_thermal_equilibrium(dissipation_rate=0.2)
        self.assertEqual(len(eq_state), 128)

        # Ensure history logged 5 steps
        self.assertEqual(len(self.engine.history), 5)

    def test_dynamic_chunking_and_macro_rotor_promotion(self):
        """Verify promotion of stabilized wave patterns to macro-rotors."""
        macro = self.engine.chunk_and_promote_macro_rotor("Rotor_Test_1", threshold_coherence=0.1)
        self.assertIsNotNone(macro)
        self.assertIsInstance(macro, MacroRotor)
        self.assertEqual(macro.rotor_id, "Rotor_Test_1")

        # Test usage count incrementing
        macro_again = self.engine.chunk_and_promote_macro_rotor("Rotor_Test_1", threshold_coherence=0.1)
        self.assertEqual(macro_again.usage_count, 2)

    def test_retrocausal_trajectory_reversibility(self):
        """Verify 100% reversible retrocausal back-tracing with Delta S ≈ 0."""
        # Perform transformations
        self.engine.apply_euler_rotation(np.pi / 3)
        self.engine.apply_log_scale_compression(base=2.0)
        self.engine.apply_force_acceleration(mass=2.0, force_amplitude=8.0)
        self.engine.apply_thermal_equilibrium(dissipation_rate=0.1)

        # Retrocausal trace back
        result = self.engine.trace_retrocausal_trajectory()
        self.assertIsInstance(result, RetrocausalFieldResult)
        self.assertGreaterEqual(result.reconstruction_fidelity, 0.99)
        self.assertLessEqual(result.reversibility_dissipation_index, 0.01)


if __name__ == "__main__":
    unittest.main()
