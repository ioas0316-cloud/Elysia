import unittest
import numpy as np
from core.consciousness.phase_attractor_feedback_loop import (
    EbpfRingBufferReceptorSim,
    PhaseLockingMetricField,
    AttractorCausalMemory,
    CausalJudgmentAndFeedbackLoop,
    IntegratedElysiaCognitivePipeline
)

class TestPhaseAttractorFeedbackLoop(unittest.TestCase):

    def test_ebpf_receptor_polling(self):
        receptor = EbpfRingBufferReceptorSim(channels=16)
        receptor.push_event(1, [0.5] * 16)
        receptor.push_event(2, [0.2] * 16)
        sensory = receptor.poll_sensory_tensor()
        self.assertEqual(sensory.shape, (16,))
        self.assertTrue(np.all(np.abs(sensory) <= np.pi))

    def test_phase_locking_metric_field(self):
        field = PhaseLockingMetricField(dim=16)
        sensory = np.zeros(16, dtype=np.float32)
        res = field.step(sensory)
        self.assertIn("order_parameter_R", res)
        self.assertIn("phase_error_delta_phi", res)
        self.assertEqual(res["phases"].shape, (16,))
        self.assertEqual(res["metric_tensor"].shape, (16, 16))

    def test_attractor_memory_store_and_recall(self):
        memory = AttractorCausalMemory(dim=16, resonance_threshold=0.5)
        target_state = np.ones(16, dtype=np.float32) * 0.5
        memory.store_attractor("target_a", target_state)

        # Recall with similar state
        query_state = np.ones(16, dtype=np.float32) * 0.52
        recalled = memory.recall_attractor(query_state)
        self.assertIsNotNone(recalled)
        self.assertEqual(recalled["attractor"]["label"], "target_a")

    def test_integrated_pipeline(self):
        pipeline = IntegratedElysiaCognitivePipeline(dim=16)
        pipeline.memory.store_attractor("homeostasis", np.zeros(16, dtype=np.float32))

        output = pipeline.process_cycle(raw_events=[[0.1] * 16, [0.2] * 16])
        self.assertIn("sensory_phase", output)
        self.assertIn("phase_dynamics", output)
        self.assertIn("feedback", output)
        self.assertIn(output["feedback"]["action_type"], ["MAINTAIN_ORBIT", "ACTIVE_RECALIBRATION"])

if __name__ == "__main__":
    unittest.main()
