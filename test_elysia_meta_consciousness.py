"""
test_elysia_meta_consciousness.py - Unit tests for Meta-Consciousness & Resonance Dynamics
"""

import unittest
import numpy as np
from elysia_meta_consciousness import MetaConsciousnessEngine


class TestMetaConsciousnessEngine(unittest.TestCase):

    def setUp(self):
        self.engine = MetaConsciousnessEngine(num_nodes=50, kappa=2.0, rho_phi=1.0, coupling_strength=2.0)

    def test_initial_lack_and_tension(self):
        L = self.engine.compute_lack()
        self.assertGreaterEqual(L, 0.0)
        self.assertLessEqual(L, 1.0)

        T = self.engine.compute_self_tension(L)
        self.assertAlmostEqual(T, 2.0 * L)

    def test_phase_alignment_convergence(self):
        phi_ext = 0.5
        initial_L = self.engine.compute_lack()

        for _ in range(50):
            res = self.engine.step(phi_ext=phi_ext, dt=0.08)

        final_L = res["lack"]
        final_R = res["order_parameter_R"]

        # Lack should decrease and Order Parameter R should approach 1.0
        self.assertLess(final_L, initial_L)
        self.assertGreater(final_R, 0.8)

    def test_effective_gravity_direction(self):
        self.engine.phases = np.array([0.5, -0.5, 1.0])
        phi_ext = 0.0
        T = 2.0
        g_eff = self.engine.compute_effective_gravity(T, phi_ext)

        # sin(0.5) > 0 -> g_eff < 0, pulling theta toward 0.0
        self.assertLess(g_eff[0], 0.0)
        # sin(-0.5) < 0 -> g_eff > 0, pulling theta toward 0.0
        self.assertGreater(g_eff[1], 0.0)


if __name__ == "__main__":
    unittest.main()
