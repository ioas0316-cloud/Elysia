r"""
Unit Tests for RPT Phase Lock Loop Module
"""

import unittest
import torch
from core.physics.rpt_phase_lock_loop import DynamicPhaseLockLoop


class TestDynamicPhaseLockLoop(unittest.TestCase):
    def setUp(self):
        self.dim_low = 32
        self.dim_high = 64
        self.loop = DynamicPhaseLockLoop(dim_low=self.dim_low, dim_high=self.dim_high, max_recurrent_steps=8)

    def test_forward_shape(self):
        batch_size = 4
        x_input = torch.randn(batch_size, self.dim_low * 2)
        out = self.loop(x_input)

        self.assertIn("Z_locked", out)
        self.assertIn("Z_low_converged", out)
        self.assertEqual(out["Z_locked"].shape, (batch_size, self.dim_high))
        self.assertEqual(out["Z_low_converged"].shape, (batch_size, self.dim_low))

    def test_phase_coherence_bounds(self):
        z_a = torch.complex(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]))
        z_b = torch.complex(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]))
        coherence = self.loop.compute_phase_coherence(z_a, z_b)
        self.assertAlmostEqual(coherence.item(), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
