r"""
test_taiji_hexagram_topology.py
===============================
Unit tests for CyclicVectorBitEngine and TaijiHexagramTopologyEngine.
"""

import math
import numpy as np
import unittest

from core.physics.cyclic_vector_bit_engine import CyclicVectorBitEngine, YinYangCyclicBit
from core.topology.taiji_hexagram_topology import TaijiHexagramTopologyEngine


class TestTaijiHexagramCosmology(unittest.TestCase):

    def test_cyclic_vector_bit(self):
        bit = YinYangCyclicBit(phase=0.0)
        self.assertAlmostEqual(bit.yin, 1.0)
        self.assertAlmostEqual(bit.yang, 0.0)

        bit.rotate(math.pi)
        self.assertAlmostEqual(bit.yin, 0.0)
        self.assertAlmostEqual(bit.yang, 1.0)

        # Dynamic vector length check
        vec = bit.dynamic_vector
        self.assertEqual(len(vec), 3)

    def test_cyclic_vector_bit_engine_breathing(self):
        engine = CyclicVectorBitEngine(num_bits=16)
        bitstream = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1])
        engine.encode_discrete_bits(bitstream)

        # Test information conservation before breathing
        loss_initial = engine.calculate_information_conservation_loss(bitstream)
        self.assertEqual(loss_initial, 0.0)

        # Step breathing dynamics
        for _ in range(5):
            report = engine.step_breathing_dynamics(dt=0.05)
            self.assertIn("taiji_balance", report)
            self.assertIn("phase_coherence", report)

        curvature = engine.extract_spectral_axis_curvature()
        self.assertEqual(len(curvature), 16)

    def test_taiji_hexagram_topology_engine(self):
        topology_engine = TaijiHexagramTopologyEngine(num_nodes=64)
        self.assertEqual(len(topology_engine.nodes), 64)

        # Test writing payload into spherical memory
        payload = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        topology_engine.write_spherical_memory(0, payload)

        # Read spherical memory back
        query_vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        best_idx, read_payload, resonance = topology_engine.read_spherical_memory(query_vec)
        self.assertGreaterEqual(best_idx, 0)
        self.assertLess(best_idx, 64)
        self.assertGreater(resonance, 0.0)

        # Run macro-micro feedback loop
        report = topology_engine.macro_micro_feedback_loop(dt=0.1)
        self.assertIn("mean_taiji_balance", report)
        self.assertIn("macro_field_energy", report)


if __name__ == "__main__":
    unittest.main()
