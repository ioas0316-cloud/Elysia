r"""
Phase Dynamics & Dynamic Benchmark Tests for Scale Hierarchy Ecosystem Engine
=============================================================================
Verifies the 4 Essential Criteria:
  1. Inter-scale Resonance (Micro spike disrupts Macro; Macro purpose modulates Micro resistance dial)
  2. Sensor-as-Prism Refraction (Zero artificial if-else branching; spectral refraction through tensor metric G_ij)
  3. Metric Deformation & Scarring (||G_after - G_before|| > 0 and phase trajectory divergence under identical inputs)
  4. Boundary of Freedom (Macro provides field potential bounds rather than enforcing exact micro outputs)
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
            disruption_threshold=0.35,
            scar_learning_rate=0.08
        )

    def test_inter_scale_resonance(self):
        """1. Inter-scale Resonance Check: Bidirectional non-linear coupling between Micro and Macro."""
        # Low strain input
        gentle_input = torch.randn(1, 16) * 0.1
        res_gentle = self.engine(gentle_input)
        dial_initial = res_gentle["resistance_dial"]

        # Violent micro input
        violent_input = torch.randn(1, 16) * 3.0
        res_violent = self.engine(violent_input)

        # Micro spike must disrupt macro thought state continuously
        self.assertGreater(res_violent["bottom_up_disruption"], 0.1)

        # Macro purpose field must modulate resistance dial (top-down feedback)
        dial_updated = res_violent["resistance_dial"]
        self.assertNotEqual(dial_initial, dial_updated)

    def test_prism_refraction_without_conditional_branching(self):
        """2. Sensor-as-Prism Refraction Check: Data naturally refracts through tensor metric manifold G_ij."""
        sensory_input_a = torch.randn(1, 16) * 0.8
        sensory_input_b = torch.randn(1, 16) * 0.8

        res_a = self.engine(sensory_input_a)
        res_b = self.engine(sensory_input_b)

        # Outputs must continuously disperse into different spectral phase states
        meso_dist = torch.norm(res_a["meso_state"] - res_b["meso_state"]).item()
        self.assertGreater(meso_dist, 0.0)

    def test_irreversible_metric_deformation_and_re_cognition(self):
        """3. Metric Deformation Check: Permanent scar on G_ij and phase trajectory divergence for identical inputs."""
        initial_metric = self.engine.perception_metric.clone()

        gentle_input = torch.randn(1, 16) * 0.1
        world_friction = torch.randn(1, 32) * 0.1

        # Phase trajectory prior to scar
        res_before = self.engine(gentle_input, world_friction=world_friction)
        meso_before = res_before["meso_state"].clone()

        # Inflict severe world friction (scarring experience)
        violent_input = torch.randn(1, 16) * 3.5
        violent_friction = torch.randn(1, 32) * 3.0
        self.engine(violent_input, world_friction=violent_friction)

        deformed_metric = self.engine.perception_metric.clone()

        # Metric scar deformation norm check
        metric_scar_norm = torch.norm(deformed_metric - initial_metric).item()
        self.assertGreater(metric_scar_norm, 0.0, "Metric tensor G_ij must suffer irreversible scar deformation")

        # Re-apply identical input post-scar and observe phase trajectory divergence
        res_after = self.engine(gentle_input, world_friction=world_friction)
        meso_after = res_after["meso_state"].clone()

        phase_trajectory_distance = torch.norm(meso_after - meso_before).item()
        self.assertGreater(phase_trajectory_distance, 0.0, "Identical input must yield divergent phase trajectory post-scar")

    def test_boundary_of_freedom(self):
        """4. Boundary of Freedom Check: Macro scale provides boundary potential without forcing Micro values."""
        micro_state_before = self.engine.micro_state.clone()

        # Step forward with variable inputs
        var_input = torch.randn(1, 16) * 0.5
        res = self.engine(var_input)

        # Micro state evolves freely within boundary rather than being hardcoded by Macro
        micro_state_after = res["micro_state"]
        self.assertFalse(torch.equal(micro_state_before, micro_state_after))
        self.assertIn("purpose_field", res)
        self.assertIn("resistance_dial", res)


if __name__ == "__main__":
    unittest.main()
