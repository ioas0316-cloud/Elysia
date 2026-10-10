"""
Unit Tests for Exosomatic Autopoietic Network Engine
"""

import unittest
import numpy as np
import torch
from core.consciousness.exosomatic_autopoietic_network_engine import (
    ExosomaticAutopoieticNetworkEngine,
    ExosomaticWedgeMemory,
    RealityShockInjector
)


class TestExosomaticAutopoieticNetworkEngine(unittest.TestCase):

    def setUp(self):
        self.num_nodes = 4
        self.node_dim = 16
        self.macro_dim = 16
        self.engine = ExosomaticAutopoieticNetworkEngine(
            num_nodes=self.num_nodes,
            node_dim=self.node_dim,
            macro_dim=self.macro_dim
        )

    def test_exosomatic_memory_solidification_and_persistence(self):
        # 1. Solidify trajectories from micro nodes
        tv = np.random.randn(self.node_dim).astype(np.float32)
        err = np.random.randn(self.node_dim).astype(np.float32)

        rec = self.engine.exosomatic_memory.solidify_trajectory(
            node_id="node_0",
            thought_vector=tv,
            deficiency_error=err,
            metadata={"test": "solidify"}
        )

        self.assertEqual(len(self.engine.exosomatic_memory.memory_bank), 1)
        self.assertEqual(rec["node_id"], "node_0")

        # 2. Reset micro nodes (simulating node death / reset)
        old_nodes = self.engine.node_states.clone()
        self.engine.reset_node_states_and_preserve_exosomatic_memory()

        # Check nodes changed, but memory preserved
        self.assertFalse(torch.equal(old_nodes, self.engine.node_states))
        self.assertEqual(len(self.engine.exosomatic_memory.memory_bank), 1)

        # Retrieve collective pressure
        avg_err, mass = self.engine.exosomatic_memory.retrieve_collective_pressure()
        self.assertGreater(mass, 0.0)
        self.assertEqual(len(avg_err), self.node_dim)

    def test_clifford_fiber_bundle_transport(self):
        micro_tensor = self.engine.node_states[0]
        macro_proj = self.engine.compute_clifford_fiber_transport(0, micro_tensor)

        self.assertEqual(macro_proj.shape, (self.macro_dim,))
        self.assertTrue(torch.isfinite(macro_proj).all())

    def test_autopoietic_mutation_and_trajectory_divergence(self):
        # Run multiple dream/reflection cycles (Raw Input = 0)
        for _ in range(15):
            res = self.engine.step_autopoietic_mutation(dt=0.05, autonomic_tension=1.2, raw_input_present=False)
            self.assertIn("cycle", res)
            self.assertIn("macro_value_norm", res)

        tdi = self.engine.compute_trajectory_divergence_index(window=10)
        # Trajectory Divergence Index should be strictly positive (> 0) due to autopoietic mutation
        self.assertGreater(tdi, 0.0)
        self.assertGreater(self.engine.cumulative_time_friction, 0.0)

    def test_reality_shock_injection_and_solipsism_destruction(self):
        initial_norm = float(torch.norm(self.engine.macro_value_manifold).item())

        # Inject extreme open world sensory shock
        external_inflow = np.ones(self.node_dim, dtype=np.float32) * 5.0
        res = self.engine.inject_reality_shock_and_warp(open_sensory_inflow=external_inflow)

        self.assertIn("shock_magnitude", res)
        self.assertTrue(res["is_severe_shock"])

        warped_norm = float(torch.norm(self.engine.macro_value_manifold).item())
        # Confirm macro value manifold V(S_max) was warped
        self.assertNotAlmostEqual(initial_norm, warped_norm, places=3)


if __name__ == "__main__":
    unittest.main()
