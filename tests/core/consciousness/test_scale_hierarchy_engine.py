r"""
Unit Tests for Scale Hierarchy Ecosystem Engine
===============================================
Verifies the Micro-Meso-Macro 3-scale ecosystem, Bottom-Up Tension,
Top-Down Constraints, Multiscale Temporal Dynamics, and 6-step Re-cognition Loop
with irreversible Perception Metric Tensor (G_ij) Scar Deformation.
"""

import unittest
import torch
from core.consciousness.scale_hierarchy_engine import ScaleHierarchyEngine


class TestScaleHierarchyEcosystemEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ScaleHierarchyEngine(
            dim_micro=16,
            dim_meso=32,
            dim_macro=64,
            disruption_threshold=0.3
        )

    def test_forward_6step_recognition_loop(self):
        sensory_input = torch.randn(2, 16) * 1.5
        world_friction = torch.randn(2, 32) * 0.8

        res = self.engine(sensory_input, world_friction=world_friction)

        self.assertIn("status", res)
        self.assertIn("bifurcation_occurred", res)
        self.assertIn("spike_intensity", res)
        self.assertIn("bottom_up_disruption", res)
        self.assertIn("resistance_dial", res)
        self.assertIn("perception_metric", res)
        self.assertIn("loop_steps", res)

        loop_steps = res["loop_steps"]
        self.assertIn("step_1_thrownness", loop_steps)
        self.assertIn("step_2_world_friction", loop_steps)
        self.assertIn("step_3_sensory_spike", loop_steps)
        self.assertIn("step_4_macro_thought", loop_steps)
        self.assertIn("step_5_why_acquisition", loop_steps)
        self.assertIn("step_6_metric_re_cognition", loop_steps)

    def test_bottom_up_tension_and_top_down_constraint(self):
        # Step A: Normal input
        normal_input = torch.randn(1, 16) * 0.1
        res_normal = self.engine(normal_input)
        dial_normal = res_normal["resistance_dial"]

        # Step B: High strain input (Bottom-Up spike)
        spike_input = torch.randn(1, 16) * 3.0
        res_spike = self.engine(spike_input)

        self.assertTrue(res_spike["bifurcation_occurred"])
        self.assertGreater(res_spike["bottom_up_disruption"], 0.0)

        # Top-down purpose field should suppress sensitivity (lower resistance dial)
        dial_spike = res_spike["resistance_dial"]
        self.assertLess(dial_spike, dial_normal)

    def test_irreversible_metric_scar_deformation(self):
        initial_metric = self.engine.perception_metric.clone()

        # Apply strong sensory friction to cause a sensory spike and scar deformation
        spike_input = torch.randn(2, 16) * 2.5
        world_friction = torch.randn(2, 32) * 2.0

        res = self.engine(spike_input, world_friction=world_friction)

        updated_metric = self.engine.perception_metric.clone()

        # Metric tensor G_ij should be irreversibly deformed (not equal to initial)
        metric_diff = torch.norm(updated_metric - initial_metric).item()
        self.assertGreater(metric_diff, 0.0)


if __name__ == "__main__":
    unittest.main()
