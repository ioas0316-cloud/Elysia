"""
tests/test_retrocausal_genesis.py

Unit tests for Retrocausal Epistemic Rupture Engine & Integrated Meta-Consciousness Engine.
"""

import unittest
import numpy as np
from core.consciousness.retrocausal_epistemic_rupture import RetrocausalEpistemicRuptureEngine
from elysia_meta_consciousness import MetaConsciousnessEngine


class TestRetrocausalEpistemicRupture(unittest.TestCase):

    def setUp(self):
        self.engine = RetrocausalEpistemicRuptureEngine(
            dim=4,
            micro_nodes=32,
            stress_threshold=2.0,
            plasticity_rate=0.2
        )

    def test_initial_state(self):
        self.assertEqual(self.engine.dim, 4)
        self.assertEqual(self.engine.micro_nodes, 32)
        np.testing.assert_array_equal(self.engine.g_metric, np.eye(4))
        self.assertAlmostEqual(self.engine.boundary_radius, 1.0)
        self.assertEqual(self.engine.dislocation_count, 0)

    def test_alterity_collision_low_stress(self):
        # Alterity wave matching actor wave phase -> low stress
        actor_wave = self.engine.compute_actor_output()
        actor_phase = np.arctan2(actor_wave[:, 0], actor_wave[:, 1])

        res = self.engine.apply_alterity_collision(actor_phase, dt=0.05)

        self.assertFalse(res["dislocated"])
        self.assertEqual(self.engine.dislocation_count, 0)
        np.testing.assert_array_almost_equal(self.engine.g_metric, np.eye(4))

    def test_alterity_collision_high_stress_and_rupture(self):
        # Alterity wave orthogonal/opposite phase -> high 1tan stress rupture
        opp_wave = np.random.uniform(-np.pi, np.pi, size=32)

        initial_anchor = self.engine.x_anchor.copy()
        res = self.engine.apply_alterity_collision(opp_wave, dt=0.05)

        self.assertTrue(res["dislocated"])
        self.assertGreater(self.engine.dislocation_count, 0)
        self.assertGreater(self.engine.residual_entropy, 0.0)
        self.assertGreater(self.engine.boundary_radius, 1.0)

        # Metric tensor g_metric should no longer be identity
        self.assertFalse(np.array_equal(self.engine.g_metric, np.eye(4)))

    def test_meta_consciousness_integration(self):
        meta_engine = MetaConsciousnessEngine(num_nodes=32)
        alterity_wave = np.random.uniform(-np.pi, np.pi, size=32)

        res = meta_engine.step(phi_ext=0.5, dt=0.05, alterity_wave=alterity_wave)

        self.assertIn("lack", res)
        self.assertIn("self_tension", res)
        self.assertIn("rupture", res)
        self.assertIn("boundary_radius", res["rupture"])


if __name__ == "__main__":
    unittest.main()
