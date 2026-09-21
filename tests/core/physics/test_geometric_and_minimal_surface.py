r"""
Unit Tests for Geometric Loss Function and Minimal Surface Regularizer
"""

import unittest
import torch
from core.physics.geometric_loss import GeometricLoss
from core.physics.minimal_surface_regularizer import MinimalSurfaceRegularizer


class TestGeometricPhysics(unittest.TestCase):
    def setUp(self):
        self.loss_fn = GeometricLoss()
        self.reg = MinimalSurfaceRegularizer()

    def test_geometric_loss_keys_and_values(self):
        psi = torch.randn(2, 16)
        z_low = torch.complex(torch.randn(2, 16), torch.randn(2, 16))
        z_fb = torch.complex(torch.randn(2, 16), torch.randn(2, 16))
        omega = torch.randn(2, 2, 16, 16)

        res = self.loss_fn(psi, z_low, z_fb, omega)
        self.assertIn("total_loss", res)
        self.assertIn("l_curvature", res)
        self.assertIn("l_phase_lock", res)
        self.assertIn("l_covariant", res)
        self.assertGreater(res["total_loss"].item(), 0.0)

    def test_minimal_surface_regularizer(self):
        psi = torch.randn(2, 16, 32)
        reg_loss = self.reg(psi)
        self.assertTrue(torch.is_tensor(reg_loss))
        self.assertFalse(torch.isnan(reg_loss))


if __name__ == "__main__":
    unittest.main()
