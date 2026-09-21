r"""
Unit Tests for 3D Level-Set & Mean Curvature Flow Engine
"""

import unittest
import torch
from core.physics.elysia_curvature_flow import ElysiaCurvatureFlow3D


class TestElysiaCurvatureFlow3D(unittest.TestCase):
    def setUp(self):
        self.flow_engine = ElysiaCurvatureFlow3D(gamma=0.1, dt=0.01, delta=0.5)

    def test_forward_shape_and_curvature(self):
        # 3D sphere Level-Set field: \Phi(r) = r - R
        grid_d, grid_h, grid_w = 16, 16, 16
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, grid_d),
            torch.linspace(-1, 1, grid_h),
            torch.linspace(-1, 1, grid_w),
            indexing='ij'
        )
        r = torch.sqrt(x**2 + y**2 + z**2)
        phi = (r - 0.5).unsqueeze(0).unsqueeze(0)  # [1, 1, 16, 16, 16]

        phi_next, H = self.flow_engine(phi)

        self.assertEqual(phi_next.shape, phi.shape)
        self.assertEqual(H.shape, phi.shape)
        self.assertFalse(torch.isnan(phi_next).any())
        self.assertFalse(torch.isnan(H).any())


if __name__ == "__main__":
    unittest.main()
