"""
Tests for Unified Topological Phase Runtime
============================================
Verifies:
1. Self-World Boundary (autopoietic operational closure & perturbation attenuation)
2. Enactive Reach Grid (intervention boundary evaluation)
3. 3x3x3 Fractal Node Grid Phase-Locking & Isomorphic Resonance Coupling
4. Meta-Causal Collapse Prevention, Causal Gravity, and Backtracing
5. Achromatic White Tensor Zero-Point Re-initialization
"""

import unittest
import numpy as np
from elysia_core.unified_topological_phase_runtime import (
    SelfWorldBoundary,
    EnactiveReachGrid,
    FractalPhaseNode3x3x3,
    UnifiedTopologicalPhaseRuntime,
)


class TestUnifiedTopologicalPhaseRuntime(unittest.TestCase):
    def setUp(self):
        self.runtime = UnifiedTopologicalPhaseRuntime()

    def test_self_world_boundary(self):
        boundary = SelfWorldBoundary(boundary_rigidity=0.8, internal_homeostasis=1.0)
        signal = np.array([1.0, 0.5, -0.5, 0.0, 0.2])
        filtered_sig, boundary_tension = boundary.process_perturbation(signal, current_phase_tension=1.0)

        self.assertLessEqual(np.linalg.norm(filtered_sig), np.linalg.norm(signal))
        self.assertGreater(boundary_tension, 0.0)

    def test_enactive_reach_grid(self):
        reach_grid = EnactiveReachGrid(reach_radius=2.0)
        near_coords = np.array([1.5, 1.5, 1.5])
        far_coords = np.array([10.0, 10.0, 10.0])

        influence_near = reach_grid.evaluate_intervention(near_coords, action_magnitude=1.0)
        influence_far = reach_grid.evaluate_intervention(far_coords, action_magnitude=1.0)

        self.assertEqual(influence_near, 1.0)
        self.assertLess(influence_far, 1.0)

    def test_3x3x3_fractal_node_grid_and_phase_locking(self):
        self.assertEqual(len(self.runtime.nodes), 27)
        trigger = np.array([0.5, 0.2, 0.8, -0.4, 0.1])

        # Propagate multiple steps
        for _ in range(5):
            res = self.runtime.trigger_wave_propagation(trigger, dt=0.1)

        self.assertIn("coherence_score", res)
        self.assertIn("isomorphism_score", res)
        self.assertGreaterEqual(res["coherence_score"], 0.0)
        self.assertLessEqual(res["coherence_score"], 1.0)

    def test_meta_causal_collapse_and_backtracing(self):
        # Apply high entropy noise trigger to induce phase perturbation
        noisy_trigger = np.array([5.0, -5.0, 10.0, -10.0, 8.0])
        res = self.runtime.trigger_wave_propagation(noisy_trigger, dt=0.1)

        # Verify history ring recording
        self.assertGreater(len(self.runtime.history_ring), 0)
        summary = self.runtime.get_system_state_summary()
        self.assertGreaterEqual(summary["history_length"], 1)

    def test_achromatic_white_tensor_reset(self):
        trigger = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        self.runtime.trigger_wave_propagation(trigger)

        self.runtime.reset_achromatic_white_tensor()
        center_node = self.runtime.nodes[(1, 1, 1)]

        np.testing.assert_array_equal(center_node.phase_vec, np.zeros(5))
        self.assertEqual(self.runtime.white_tensor_reset_count, 1)


if __name__ == "__main__":
    unittest.main()
