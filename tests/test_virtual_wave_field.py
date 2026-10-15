"""
test_virtual_wave_field.py

Unit tests for Virtual Wave Field core implementation (`core/physics/virtual_wave_field.py`).
Verifies bit pattern sequence texture mapping, interference gates, macro-rotor chunking,
and 0D->1D->2D->3D spatiotemporal dimensional expansion.
"""

import unittest
import numpy as np
from core.physics.virtual_wave_field import (
    BitPatternPhasor,
    InterferenceGate,
    MacroRotor,
    SpatiotemporalWaveField
)


class TestVirtualWaveField(unittest.TestCase):

    def test_bit_pattern_sequence_texture(self):
        """
        Tests that 0110 and 1001 have equal density (0.5) but different spatial dispersion
        and dominant phases, proving sequence texture is captured.
        """
        phasor_condensed = BitPatternPhasor([0, 1, 1, 0])  # Condensed in middle
        phasor_dispersed = BitPatternPhasor([1, 0, 0, 1])  # Dispersed at ends

        self.assertEqual(phasor_condensed.density, 0.5)
        self.assertEqual(phasor_dispersed.density, 0.5)

        # Centroid is middle (1.5) for both
        self.assertAlmostEqual(phasor_condensed.centroid, 1.5)
        self.assertAlmostEqual(phasor_dispersed.centroid, 1.5)

        # Dispersion of dispersed pattern must be greater than condensed pattern
        self.assertGreater(phasor_dispersed.dispersion, phasor_condensed.dispersion)

        # Dominant phase should differ due to pattern structure
        self.assertNotAlmostEqual(phasor_condensed.dominant_phase, phasor_dispersed.dominant_phase)

    def test_interference_gate_superposition(self):
        """
        Tests constructive and destructive interference gate logic.
        """
        gate = InterferenceGate(resonance_threshold=0.75)

        p1 = BitPatternPhasor([1, 1, 0, 0])
        p2_aligned = BitPatternPhasor([1, 1, 0, 0])
        p3_opposing = BitPatternPhasor([0, 0, 1, 1])

        # Identical patterns -> Constructive interference & Phase Lock
        res_aligned = gate.evaluate(p1, p2_aligned)
        self.assertGreaterEqual(res_aligned["resonance_index"], 0.75)
        self.assertTrue(res_aligned["is_phase_locked"])
        self.assertAlmostEqual(res_aligned["phase_coherence"], 1.0, places=4)

        # Opposing patterns -> Destructive / lower resonance
        res_opposing = gate.evaluate(p1, p3_opposing)
        self.assertLess(res_opposing["resonance_index"], res_aligned["resonance_index"])

    def test_macro_rotor_aggregation(self):
        """
        Tests MacroRotor aggregation of micro-phasors into higher-level harmonic state.
        """
        macro = MacroRotor(name="MacroAlpha")
        p1 = BitPatternPhasor([1, 1, 0, 0])
        p2 = BitPatternPhasor([1, 1, 1, 0])

        macro.add_micro_phasor(p1)
        macro.add_micro_phasor(p2)

        self.assertEqual(len(macro.micro_phasors), 2)
        self.assertGreater(macro.amplitude, 0.0)

        macro_beta = MacroRotor(name="MacroBeta")
        p3 = BitPatternPhasor([0, 0, 1, 1])
        macro_beta.add_micro_phasor(p3)

        coupled_amp, alignment = macro.interfere_with(macro_beta)
        self.assertIsInstance(coupled_amp, float)
        self.assertIsInstance(alignment, float)

    def test_dimensional_expansion(self):
        """
        Tests 0D (Point) -> 1D (Line) -> 2D (Plane) -> 3D (Spatiotemporal) dimensional expansion.
        """
        field = SpatiotemporalWaveField(field_dim=16)

        # 1. Point 0D: Transition delta
        phase_shift = field.project_point_transition(0, 1)
        self.assertAlmostEqual(phase_shift, np.pi / 2.0)

        # 2. Line 1D: Sequence trajectory
        seq_x = [0, 1, 1, 0, 1, 0, 0, 1]
        seq_y = [1, 0, 0, 1, 0, 1, 1, 0]
        line_x = field.project_line_sequence(seq_x)
        line_y = field.project_line_sequence(seq_y)

        self.assertEqual(len(line_x), len(seq_x))
        self.assertEqual(len(line_y), len(seq_y))

        # 3. Plane 2D: Spectral interface weaving
        plane = field.weave_plane_interface(line_x, line_y)
        self.assertEqual(plane.shape, (16, 16))

        # 4. Spatiotemporal Structure: Global coherence & Phase locking
        m1 = MacroRotor("Rotor1", [BitPatternPhasor(seq_x)])
        m2 = MacroRotor("Rotor2", [BitPatternPhasor(seq_x)])  # Identical -> High order parameter
        field.register_macro_rotor(m1)
        field.register_macro_rotor(m2)

        state = field.synthesize_spatiotemporal_structure()
        self.assertGreaterEqual(state["order_parameter_r"], 0.8)
        self.assertTrue(state["is_fully_phase_locked"])


if __name__ == "__main__":
    unittest.main()
