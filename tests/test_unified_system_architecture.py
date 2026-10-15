import unittest
import numpy as np

from elysia_core.unified_system_architecture import (
    ComplexPhasor,
    PhasorField,
    PrismBifurcationOperator,
    DynamicMacroRotorStack,
    MacroRotor,
    RetrocausalExecutionLoop,
    ReversibleWaveTransformation,
    FractalResonanceUnit,
    UnifiedSystemArchitecture,
)


class TestUnifiedSystemArchitecture(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

    def test_stage1_complex_phasor_and_field(self):
        # Test ComplexPhasor representation and math
        p1 = ComplexPhasor(amplitude=2.0, phase=np.pi / 4, frequency=2.0)
        p2 = ComplexPhasor(amplitude=1.0, phase=np.pi / 4, frequency=2.0)

        self.assertAlmostEqual(p1.amplitude, 2.0)
        self.assertAlmostEqual(p1.phase, np.pi / 4)

        # Test interference addition
        p_sum = p1 + p2
        self.assertAlmostEqual(p_sum.amplitude, 3.0)
        self.assertAlmostEqual(p_sum.phase, np.pi / 4)

        # Test PhasorField creation and coherence calculation
        shape = (4, 4)
        field_coherent = PhasorField(shape, field=np.ones(shape, dtype=np.complex128))
        coherence = field_coherent.calculate_coherence()
        self.assertAlmostEqual(coherence, 1.0)

        # Incoherent field test
        phases = np.linspace(0, 2 * np.pi, 16, endpoint=False).reshape(shape)
        field_incoherent = PhasorField(shape, field=np.exp(1j * phases))
        self.assertLess(field_incoherent.calculate_coherence(), 0.1)

    def test_stage2_prism_bifurcation_operator(self):
        root_field = PhasorField((3, 3), field=2.0 * np.ones((3, 3), dtype=np.complex128))
        prism = PrismBifurcationOperator(num_phases=3, threshold=0.1)
        sub_spectra = prism.refract(root_field, root_id="test_root")

        self.assertEqual(len(sub_spectra), 3)
        for idx, (branch_field, tag) in enumerate(sub_spectra):
            self.assertEqual(tag.parent_id, "test_root")
            self.assertEqual(tag.branch_index, idx)
            self.assertTrue(tag.threshold_passed)
            self.assertEqual(branch_field.shape, (3, 3))

    def test_stage3_dynamic_macro_rotor_stack(self):
        memory = DynamicMacroRotorStack(max_layers=3, chunking_coherence_threshold=0.6)
        coherent_field = PhasorField((4, 4), field=np.ones((4, 4), dtype=np.complex128))

        # First chunking step
        promoted = memory.process_and_chunk(coherent_field)
        self.assertEqual(len(promoted), 1)
        self.assertEqual(promoted[0].layer_level, 1)

        # Process additional fields to trigger Level 2 promotion
        memory.process_and_chunk(coherent_field)
        memory.process_and_chunk(coherent_field)

        self.assertGreaterEqual(len(memory.layers[1]), 3)
        self.assertGreaterEqual(len(memory.layers[2]), 1)

    def test_stage4_retrocausal_loop_and_self_correction(self):
        retro_loop = RetrocausalExecutionLoop(entropy_tolerance=1e-6)
        initial_field = PhasorField((4, 4), field=np.random.uniform(0.5, 1.5, size=(4, 4)) * np.exp(1j * np.random.uniform(0, 2 * np.pi, size=(4, 4))))

        rot_transform = RetrocausalExecutionLoop.create_phase_rotation_transform(np.pi / 3)
        scale_transform = RetrocausalExecutionLoop.create_amplitude_scaling_transform(2.5)
        transforms = [rot_transform, scale_transform]

        # Forward execution
        step1 = rot_transform.apply_forward(initial_field)
        final_field = scale_transform.apply_forward(step1)

        # Retrocausal trace validation
        valid, entropy_loss, recovered = retro_loop.retrocausal_trace(initial_field, final_field, transforms)

        self.assertTrue(valid)
        self.assertLess(entropy_loss, 1e-6)
        np.testing.assert_allclose(recovered.field, initial_field.field, rtol=1e-5, atol=1e-5)

        # Geometric phase self-correction test (Moire Overlap Principle)
        phase_distortion = np.random.normal(0, 0.3, size=(4, 4))
        distorted_field = initial_field.rotate(phase_distortion)

        rectified, mismatch_map, global_distortion = retro_loop.detect_and_rectify_phase_errors(
            distorted_field, initial_field
        )

        self.assertGreater(global_distortion, 0.0)
        np.testing.assert_allclose(rectified.field, initial_field.field, rtol=1e-5, atol=1e-5)

    def test_stage5_fractal_resonance_scaling(self):
        fractal_unit = FractalResonanceUnit(unit_id="test_fractal", dimension=3, depth=2)
        initial_coherence = fractal_unit.local_field.calculate_coherence()

        # Execute wave collapse
        final_coherence = fractal_unit.wave_collapse(coupling_strength=0.9, max_steps=10)

        self.assertGreaterEqual(final_coherence, initial_coherence)
        macro_rotor = fractal_unit.aggregate_macro_phasor()
        self.assertIsInstance(macro_rotor, ComplexPhasor)

    def test_full_unified_system_architecture_cycle(self):
        usa = UnifiedSystemArchitecture(num_prism_phases=3, fractal_depth=2)
        usa.initialize_root_field((4, 4), random_seed=42)

        summary = usa.execute_full_cycle()

        self.assertIn("Stage1", summary)
        self.assertIn("Stage2", summary)
        self.assertIn("Stage3", summary)
        self.assertIn("Stage4", summary)
        self.assertIn("Stage5", summary)

        self.assertTrue(summary["Stage4"]["retrocausal_valid"])
        self.assertLess(summary["Stage4"]["entropy_loss_Delta_S"], 1e-6)


if __name__ == "__main__":
    unittest.main()
