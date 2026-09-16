"""
Unit Tests for Causal DNA Helix Engine
=====================================
"""

import math
import torch
import unittest
from core.topology.causal_dna_helix_engine import (
    LogarithmicMapper,
    DoubleHelixTopology,
    CausalAtomicTensor,
    MultiLODDialIndexer,
    MultiRotorBufferManager,
    ContextPlaneTransformer,
    WordPhaseManifold,
    CausalDNAHelixEngine
)


class TestCausalDNAHelixEngine(unittest.TestCase):

    def setUp(self):
        self.log_mapper = LogarithmicMapper()
        self.helix_topo = DoubleHelixTopology(dimension=16)
        self.causal_atomic = CausalAtomicTensor()
        self.lod_indexer = MultiLODDialIndexer(self.log_mapper)
        self.engine = CausalDNAHelixEngine(vram_budget_mb=100)

    def test_logarithmic_mapper(self):
        a = torch.tensor([10.0, 100.0, 1000.0], dtype=torch.float32)
        b = torch.tensor([2.0, 5.0, 10.0], dtype=torch.float32)

        # log(A * B) == log A + log B (with high accuracy in log space)
        log_a = self.log_mapper.to_log_space(a)
        log_b = self.log_mapper.to_log_space(b)
        log_ab = self.log_mapper.to_log_space(a * b)

        # In log space: log(x + eps) ~ log(x) for large x
        torch.testing.assert_close(log_a + log_b, log_ab, rtol=1e-3, atol=1e-3)

        # Octave quantization
        octaves = self.log_mapper.quantize_octaves(log_ab, num_octaves=4)
        self.assertEqual(octaves.shape, a.shape)
        self.assertTrue(torch.all(octaves >= 0) and torch.all(octaves < 4))

    def test_double_helix_topology(self):
        signal = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32)

        wave_0, shell_0 = self.helix_topo.generate_helix_pair(signal, theta=0.0)
        wave_pi, shell_pi = self.helix_topo.generate_helix_pair(signal, theta=math.pi / 2)

        # Invariant shell axis must remain identical regardless of phase angle rotation
        torch.testing.assert_close(shell_0, shell_pi)

        # Wave axis rotates continuously
        self.assertEqual(wave_0.shape, signal.shape)
        self.assertEqual(wave_pi.shape, signal.shape)
        self.assertFalse(torch.allclose(wave_0, wave_pi))

    def test_causal_atomic_tensor(self):
        # Quarter-phase boundary transition test
        s_0 = self.causal_atomic.get_atomic_state_by_phase(0.0)                  # Direct Action (1,0)
        s_pi2 = self.causal_atomic.get_atomic_state_by_phase(math.pi / 2)         # Phase-Lock (1,1)
        s_pi = self.causal_atomic.get_atomic_state_by_phase(math.pi)             # Feedback Action (0,1)
        s_3pi2 = self.causal_atomic.get_atomic_state_by_phase(3 * math.pi / 2)    # Orthogonality (0,0)

        torch.testing.assert_close(s_0, self.causal_atomic.S_DIRECT)
        torch.testing.assert_close(s_pi2, self.causal_atomic.S_LOCK)
        torch.testing.assert_close(s_pi, self.causal_atomic.S_FEEDBACK)
        torch.testing.assert_close(s_3pi2, self.causal_atomic.S_ORTH)

        # Kronecker product expansion (2x2 -> 4x4)
        kron_4x4 = self.causal_atomic.kronecker_expand([s_pi2, s_0])
        self.assertEqual(kron_4x4.shape, (4, 4))

    def test_multi_lod_dial_indexer(self):
        matrix_16x16 = torch.ones((16, 16), dtype=torch.float32)

        sliced_lod0 = self.lod_indexer.slice_by_lod(matrix_16x16, lod_level=0)
        sliced_lod1 = self.lod_indexer.slice_by_lod(matrix_16x16, lod_level=1)
        sliced_lod2 = self.lod_indexer.slice_by_lod(matrix_16x16, lod_level=2)

        self.assertEqual(sliced_lod0.shape, (16, 16))
        self.assertEqual(sliced_lod1.shape, (8, 8))
        self.assertEqual(sliced_lod2.shape, (4, 4))

    def test_context_plane_transformer_and_word_manifold(self):
        word_builder = WordPhaseManifold(self.engine)

        # Phase-locked syllables for non-zero matrix multiplication
        han_thetas = (math.pi / 2, math.pi / 2)
        geul_thetas = (math.pi / 2, math.pi / 2)

        res_noun = word_builder.construct_word_manifold([han_thetas, geul_thetas], context_type="NOUN_FIELD")
        res_verb = word_builder.construct_word_manifold([han_thetas, geul_thetas], context_type="VERB_FIELD")

        self.assertEqual(res_noun["word_manifold"].shape, (4, 4))
        self.assertEqual(res_verb["word_manifold"].shape, (4, 4))
        self.assertFalse(torch.allclose(res_noun["word_manifold"], res_verb["word_manifold"]))

    def test_causal_dna_helix_engine_pipeline(self):
        raw_signal = torch.randn(16, dtype=torch.float32)
        res = self.engine.process_dial_rotation(
            node_id="test_node",
            raw_signal=raw_signal,
            dial_theta=math.pi / 3,
            lod_level=1
        )

        self.assertEqual(res["node_id"], "test_node")
        self.assertEqual(res["causal_matrix"].shape, (2, 2))
        self.assertEqual(res["wave_axis"].shape, raw_signal.shape)


if __name__ == "__main__":
    unittest.main()
