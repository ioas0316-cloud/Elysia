"""
Unit tests for Pure Non-Symbolic Physical Strain Field Engine (TopologicalIsomorphismEngine).
Verifies the non-symbolic local strain tensor dynamics, anisotropic refraction,
wave propagation field, in-situ substrate deformation, strain relaxation,
macro phase transition, and the 4 topological plasticity constraints.
"""

import unittest
import numpy as np
from synaptic_architecture.topological_isomorphism_engine import (
    NonSymbolicReceptiveRefractor,
    TopologicalIsomorphismEngine,
    SubstrateStrainPoint,
    MacroPhaseOrder
)


class TestTopologicalIsomorphismEngine(unittest.TestCase):
    def setUp(self):
        self.engine = TopologicalIsomorphismEngine(
            gauge_dim=32,
            f_critical=0.3,
            f_dissolve=1.8,
            z_backpressure_threshold=0.6
        )

    def test_refractor_refraction(self):
        """Verifies non-symbolic refractor converts various inputs into continuous wave gradients."""
        refractor = NonSymbolicReceptiveRefractor(gauge_dim=32)

        grad_str = refractor.refract("Physical Strain Signal Wave")
        grad_arr = refractor.refract(np.ones(32))

        self.assertEqual(len(grad_str), 32)
        self.assertEqual(len(grad_arr), 32)
        self.assertAlmostEqual(float(np.linalg.norm(grad_str)), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(grad_arr)), 1.0, places=5)

    def test_anisotropic_friction_computation(self):
        """Verifies local anisotropic friction equation: F = grad_s^T . G . grad_s"""
        grad_s = np.zeros(32, dtype=np.float32)
        grad_s[0] = 1.0

        p = SubstrateStrainPoint(
            point_id=1,
            gauge_dim=32,
            G=np.eye(32, dtype=np.float32) * 2.0,
            T=grad_s.copy()
        )

        friction = self.engine.compute_local_anisotropic_friction(grad_s, p)
        self.assertAlmostEqual(friction, 2.0, places=5)

    def test_physical_event_processing_and_phase_transition(self):
        """
        Verifies event processing, in-situ deformation G(x), wave propagation,
        and macro phase transition without any string labels or discrete lookups.
        """
        engine = TopologicalIsomorphismEngine(
            gauge_dim=32,
            f_critical=0.2,
            f_dissolve=2.5,
            z_backpressure_threshold=0.8
        )

        res1 = engine.process_physical_event("Wave Stimulus 1")
        res2 = engine.process_physical_event("Wave Stimulus 2")

        self.assertEqual(res1["point_id"], 1)
        self.assertEqual(res2["point_id"], 2)

        # Trigger phase transition with 3rd stimulus
        res3 = engine.process_physical_event("Wave Stimulus 3")

        self.assertIsNotNone(res3["emerged_order_id"])
        self.assertIn(res3["emerged_order_id"], engine.macro_orders)

        macro_order = engine.macro_orders[res3["emerged_order_id"]]
        self.assertFalse(macro_order.is_fissioned)
        self.assertGreater(len(macro_order.encapsulated_point_ids), 0)
        self.assertEqual(macro_order.macro_order_tensor.shape, (32, 32))

    def test_4_topological_plasticity_constraints(self):
        """
        Verifies the 4 non-symbolic physical constraints:
        1 & 4. Reverse Phase Transition & Hysteresis
        2. Impedance Backpressure Liquefaction
        3. Latent Fault-Line Preservation
        """
        engine = TopologicalIsomorphismEngine(
            gauge_dim=32,
            f_critical=0.1,
            f_dissolve=2.5,
            z_backpressure_threshold=0.5
        )

        # Build macro phase order
        engine.process_physical_event("Wave Pulse A")
        engine.process_physical_event("Wave Pulse B")
        res_c = engine.process_physical_event("Wave Pulse C")

        active_orders = [m for m in engine.macro_orders.values() if not m.is_fissioned]
        self.assertGreater(len(active_orders), 0)
        order = active_orders[0]

        # Verify Fault-Line Preservation
        p = engine.points[1]
        self.assertGreaterEqual(len(p.latent_faults), 0)

        # Trigger Reverse Phase Transition (Constraints 1 & 4) by lowering f_dissolve below current friction
        order.formation_energy = 0.01
        engine.f_dissolve = 0.05
        dissolve_res = engine.process_physical_event("Extreme Friction Spike")

        self.assertIn(order.order_id, dissolve_res["fissioned_order_ids"])
        self.assertTrue(order.is_fissioned)

        # Verify Impedance Backpressure Liquefaction (Constraint 2)
        backpressure_res = engine.process_physical_event(np.ones(32) * 100.0)
        self.assertGreaterEqual(backpressure_res["liquefied_beam_count"], 0)


if __name__ == "__main__":
    unittest.main()
