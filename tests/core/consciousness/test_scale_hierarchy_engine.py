r"""
Unit Tests for Scale Hierarchy Engine
"""

import unittest
import torch
from core.consciousness.scale_hierarchy_engine import ScaleHierarchyEngine


class TestScaleHierarchyEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ScaleHierarchyEngine(dim_l1=16, dim_l2=32, dim_l3=64, dim_l4=128)

    def test_forward_execution(self):
        sensory_input = torch.randn(2, 32)
        res = self.engine(sensory_input)

        self.assertIn("status", res)
        self.assertIn("bifurcation_occurred", res)
        self.assertIn("action_wave_emitted", res)
        self.assertIn("divergence_origin_scale", res)

    def test_transparent_filtering(self):
        # Noise input with near-zero resonance
        sensory_noise = torch.randn(2, 32) * 0.01
        is_transparent, score = self.engine.check_transparent_filtering(sensory_noise)
        self.assertTrue(isinstance(is_transparent, bool))
        self.assertTrue(0.0 <= score <= 1.0)


if __name__ == "__main__":
    unittest.main()
