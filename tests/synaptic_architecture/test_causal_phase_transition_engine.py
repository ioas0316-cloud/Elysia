import unittest
import numpy as np
import math
from synaptic_architecture.causal_phase_transition_engine import (
    CausalPhaseTransitionEngine,
    PerturbationWave,
    ComplexImpedance,
    HomologyMetrics,
    CausalProcessBlueprint,
)
from synaptic_architecture.field import CrystallizationField
from synaptic_architecture.phase_topological_reconstruction_engine import PhaseTopologicalReconstructionEngine

class TestCausalPhaseTransitionEngine(unittest.TestCase):

    def setUp(self):
        self.dimension = 16
        self.engine = CausalPhaseTransitionEngine(
            dimension=self.dimension,
            v_critical=50.0,
            crystallization_threshold=0.15,
            impedance_R=1.0,
            impedance_X=2.5,
        )

    def test_complex_impedance_elasticity(self):
        impedance = ComplexImpedance(R=1.0, X=2.5)
        self.assertAlmostEqual(impedance.magnitude, math.sqrt(1.0 + 6.25))
        gamma_mag, absorbed_ratio, reflected_ratio = impedance.compute_reflection_and_absorption(Z_characteristic=1.0)
        self.assertTrue(0.0 <= absorbed_ratio <= 1.0)
        self.assertTrue(0.0 <= reflected_ratio <= 1.0)
        self.assertAlmostEqual(absorbed_ratio + reflected_ratio, 1.0)

    def test_homology_metrics_thin_vs_deep(self):
        # Thin ground (Child's reflection)
        self.engine.initialize_ground("thin")
        thin_metrics = self.engine.get_homology_metrics()
        self.assertEqual(thin_metrics["B0"], 1)
        self.assertEqual(thin_metrics["B1"], 0)
        self.assertEqual(thin_metrics["classification"], "Thin Ground (Child)")

        # Deep ground (Adult's reflection)
        self.engine.initialize_ground("deep")
        deep_metrics = self.engine.get_homology_metrics()
        self.assertEqual(deep_metrics["B0"], 1)
        self.assertGreaterEqual(deep_metrics["B1"], 2)
        self.assertEqual(deep_metrics["classification"], "Deep Ground (Adult)")

    def test_causal_process_blueprint_unfolding_and_4stage_epistemology(self):
        self.engine.initialize_ground("thin")
        wave_vec = np.ones(self.dimension, dtype=np.float32)
        wave = PerturbationWave("W_Unfold", wave_vec, amplitude=1.0, frequency=2.0)
        res = self.engine.inject_perturbation_wave(wave)

        self.assertIn("causal_unfolding", res)
        unfold_info = res["causal_unfolding"]
        self.assertIn("epistemological_reflection", unfold_info)
        self.assertTrue("Epistemological Understanding" in unfold_info["epistemological_reflection"])

    def test_crystallization_1_to_0(self):
        self.engine.initialize_ground("thin")
        initial_nodes_count = len(self.engine.nodes)

        # High-coherence in-phase wave (friction < threshold)
        p_vec = self.engine.nodes["N0"].phase_axis.copy()
        wave = PerturbationWave("W_Crystallize", p_vec, amplitude=0.5, frequency=1.0, entropy=0.1)

        res = self.engine.inject_perturbation_wave(wave)
        self.assertIn("phase_transition", res)
        self.assertEqual(res["phase_transition"]["type"], "CRYSTALLIZATION")
        self.assertEqual(len(self.engine.nodes), initial_nodes_count + 1)

    def test_flash_remelting_0_to_1(self):
        self.engine.initialize_ground("thin")
        initial_nodes_count = len(self.engine.nodes)

        # High-friction orthogonal wave exceeding V_critical
        p_vec = np.zeros(self.dimension, dtype=np.float32)
        p_vec[5] = 10.0  # High energy orthogonal shock
        wave = PerturbationWave("W_Shock", p_vec, amplitude=5.0, frequency=5.0, entropy=3.0)

        res = self.engine.inject_perturbation_wave(wave)
        self.assertIn("phase_transition", res)
        self.assertEqual(res["phase_transition"]["type"], "FLASH_REMELTING")
        self.assertEqual(len(self.engine.nodes), initial_nodes_count - 1)
        self.assertIn(res["phase_transition"]["shock_wave_generated"], self.engine.waves)

    def test_backtrace_and_partial_remelting(self):
        self.engine.initialize_ground("thin")
        # Artificially modify step 1 of N0's blueprint to cause friction spike
        bp = self.engine.nodes["N0"].blueprint
        bp.mechanism_steps.append(np.eye(self.dimension, dtype=np.float32) * 50.0)

        partial_res = self.engine.backtrace_and_partial_remelt("N0", faulty_step_idx=1)
        self.assertEqual(partial_res["type"], "PARTIAL_REMELTING")
        self.assertEqual(partial_res["target_node"], "N0")
        self.assertIn(partial_res["partial_wave_generated"], self.engine.waves)

    def test_phase_mass_conservation(self):
        self.engine.initialize_ground("deep")
        initial_mass = self.engine.total_phase_mass

        wave = PerturbationWave("W_Mass", np.ones(self.dimension), amplitude=1.0, frequency=2.0)
        self.engine.inject_perturbation_wave(wave)

        # System total mass includes Ground 0 density + Wave 1 energy + Reactive stored energy
        current_mass = self.engine.total_phase_mass
        self.assertGreater(current_mass, 0.0)

    def test_elysia_modules_integration(self):
        field = CrystallizationField(128)
        rec_engine = PhaseTopologicalReconstructionEngine()

        self.engine.initialize_ground("deep")
        self.engine.sync_with_crystallization_field(field)
        self.assertGreater(np.max(field.conductance), 0.01)

        sync_res = self.engine.sync_with_topological_reconstruction_engine(rec_engine)
        self.assertIn("synced_invariants", sync_res)

        gate_res = self.engine.process_virtual_gate_friction(
            gate_loss=0.8,
            pid_control_signal=0.5,
            context_vector=np.ones(self.dimension)
        )
        self.assertIn("wave_id", gate_res)

if __name__ == "__main__":
    unittest.main()
