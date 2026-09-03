import unittest
import numpy as np
import math

from synaptic_architecture.self_codification_engine import (
    FilteringLens,
    SelfCodificationRecord,
    SelfCodificationEngine,
)


class TestSelfCodificationEngine(unittest.TestCase):
    def setUp(self):
        self.dim = 16
        self.engine = SelfCodificationEngine(
            dimension=self.dim,
            v_critical=20.0,
            crystallization_threshold=0.3,
            lens_vth=0.1,
        )

    def test_initialization_ground(self):
        # Verify base anchor ground coordinates initialization
        metrics = self.engine.phase_engine.get_homology_metrics()
        self.assertGreaterEqual(metrics["V"], 3)
        self.assertEqual(len(self.engine.codification_history), 0)

    def test_filtering_lens_translation(self):
        lens = FilteringLens(dimension=self.dim, threshold_vth=0.5)
        # Subthreshold signal
        weak_signal = np.ones(self.dim, dtype=np.float32) * 0.01
        refracted, friction, node_id = lens.translate_raw_signal(weak_signal, self.engine.phase_engine.nodes)
        self.assertEqual(node_id, "Subthreshold_Gated")
        self.assertEqual(friction, 0.0)

        # Above threshold signal
        strong_signal = np.ones(self.dim, dtype=np.float32) * 2.0
        refracted, friction, node_id = lens.translate_raw_signal(strong_signal, self.engine.phase_engine.nodes)
        self.assertIn(node_id, self.engine.phase_engine.nodes)
        self.assertGreaterEqual(friction, 0.0)

    def test_process_external_stimulus_crystallization(self):
        # Create a signal almost aligned with existing GroundNode N0 phase axis
        n0_phase = self.engine.phase_engine.nodes["N0"].phase_axis
        resonant_signal = n0_phase * 1.5

        initial_node_count = len(self.engine.phase_engine.nodes)
        res = self.engine.process_external_stimulus(resonant_signal, wave_id="Resonant_Wave_1")

        self.assertIn("wave_id", res)
        self.assertIn("codification_record", res)
        self.assertGreater(len(self.engine.codification_history), 0)

        # Metacognitive history check
        history = self.engine.backtrace_metacognitive_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["trigger_wave"], "Resonant_Wave_1")
        self.assertIn("narrative", history[0])

    def test_process_external_stimulus_remelting(self):
        # High orthogonal wave producing friction > v_critical
        orthogonal_signal = np.zeros(self.dim, dtype=np.float32)
        orthogonal_signal[5] = 100.0  # High magnitude orthogonal impulse

        res = self.engine.process_external_stimulus(orthogonal_signal, wave_id="Shockwave_Impulse")
        history = self.engine.backtrace_metacognitive_history()

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["trigger_wave"], "Shockwave_Impulse")
        self.assertIn("Self-Codification Event", history[0]["narrative"])


if __name__ == "__main__":
    unittest.main()
