"""
Unit and Integration Tests for Self-Supervised Structural Adaptation (SSA) Engine & Simulator
=================================================================================================
"""

import math
import unittest
import numpy as np

from synaptic_architecture.self_supervised_structural_adaptation import (
    ContinuousSensoryField,
    InternalManifoldState,
    MultiLensProjectionOperator,
    SelfSupervisedStructuralAdaptationEngine
)
from simulators.ssa_continuous_manifold_sim import SSAContinuousManifoldSimulator


class TestSelfSupervisedStructuralAdaptation(unittest.TestCase):

    def setUp(self):
        self.grid_size = 12
        self.engine = SelfSupervisedStructuralAdaptationEngine(
            grid_size=self.grid_size,
            kuramoto_coupling_K=1.5,
            learning_rate_phase=0.2,
            learning_rate_metric=0.05,
            resonance_threshold=0.1
        )

    def test_continuous_sensory_field_generation(self):
        field = ContinuousSensoryField.generate_wave(spatial_dim=self.grid_size, t=0.5, freq=1.2)
        self.assertEqual(field.field_matrix.shape, (self.grid_size, self.grid_size))
        self.assertTrue(np.all(field.field_matrix >= -1.0) and np.all(field.field_matrix <= 1.0))

    def test_internal_manifold_initialization(self):
        manifold = InternalManifoldState.initialize(grid_size=self.grid_size)
        self.assertEqual(manifold.metric_g.shape, (self.grid_size, self.grid_size, 2, 2))
        self.assertEqual(manifold.phase_theta.shape, (self.grid_size, self.grid_size))

        # Check initial identity metric g_ij = delta_ij
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                np.testing.assert_array_almost_equal(manifold.metric_g[i, j], np.eye(2))

    def test_multi_lens_projection(self):
        operator = MultiLensProjectionOperator(spatial_dim=self.grid_size)
        sensory_wave = ContinuousSensoryField.generate_wave(spatial_dim=self.grid_size).field_matrix
        internal_phase = np.zeros((self.grid_size, self.grid_size))

        p_math = operator.project_mathematical(sensory_wave)
        p_phys = operator.project_physical(sensory_wave)
        p_sem = operator.project_semantic(sensory_wave)

        self.assertEqual(p_math.shape, (self.grid_size, self.grid_size))
        self.assertEqual(p_phys.shape, (self.grid_size, self.grid_size))
        self.assertEqual(p_sem.shape, (self.grid_size, self.grid_size))

        best_proj, active_lens = operator.project_best_lens(sensory_wave, internal_phase)
        self.assertIn(active_lens, ["mathematical", "physical", "semantic"])

    def test_ricci_curvature_and_friction_computation(self):
        ricci_tensor, scalar_curvature = self.engine.compute_ricci_tensor(self.engine.manifold.metric_g)
        self.assertEqual(ricci_tensor.shape, (self.grid_size, self.grid_size, 2, 2))
        self.assertEqual(scalar_curvature.shape, (self.grid_size, self.grid_size))

        projected = np.zeros((self.grid_size, self.grid_size))
        friction, grad, hessian = self.engine.compute_causal_phase_friction(projected, scalar_curvature)

        self.assertGreaterEqual(friction, 0.0)
        self.assertEqual(grad.shape, (self.grid_size, self.grid_size))
        self.assertEqual(hessian.shape, (self.grid_size, self.grid_size, 2, 2))

    def test_ssa_engine_step_execution(self):
        field = ContinuousSensoryField.generate_wave(spatial_dim=self.grid_size, t=0.1)
        res = self.engine.step(field, time_delta=0.1)

        self.assertIn("friction_energy", res)
        self.assertIn("active_lens", res)
        self.assertIn("carved_volumes_count", res)
        self.assertGreaterEqual(res["carved_volumes_count"], 0)

    def test_ssa_simulator_run(self):
        simulator = SSAContinuousManifoldSimulator(grid_size=10)
        history = simulator.run_simulation(num_steps=5, verbose=False)

        self.assertEqual(len(history), 5)
        self.assertEqual(history[0]["step"], 1)
        self.assertEqual(history[-1]["step"], 5)
        self.assertIn("friction_energy", history[0])


if __name__ == "__main__":
    unittest.main()
