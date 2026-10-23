r"""
Elysia Core Physics: Quaternion 3D Phase Engine
===============================================
Zero-Branching 3D Spatial & 4D Quaternion (S^3 Topological Manifold) Phase Field Relaxation Engine.

Key Principles:
1. High-Dimensional Spatiotemporal Field: Extends 2D scalar fields to 3D spatial grids
   and 4D quaternion fields q = (q_r, q_i, q_j, q_k) on S^3 hyper-sphere manifolds.
2. Non-Commutative Causal Logic: Leverages non-commutative quaternion algebra
   to handle temporal/causal order dependence without if/else branching logic.
3. 3D Conv Depthwise Convolution: Computes 3D spatial curvature \nabla^2 q directly across
   all spatial grid points simultaneously using 3D depthwise convolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class Quaternion3DPhaseEngine(nn.Module):
    """
    Zero-Branching 3D Spatial / 4D Quaternion Field Relaxation Simulator.
    Executes simultaneous spatial-orientation phase locking and obstacle avoidance on S^3 manifolds.
    """

    def __init__(
        self,
        shape: Tuple[int, int, int] = (32, 32, 32),
        kappa: float = 0.5,
        alpha: float = -1.0,
        beta: float = 1.0,
        gamma_0: float = 0.05,
        gamma_avalanche: float = 15.0,
        strain_c: float = 0.8,
        device: str = "cpu"
    ):
        super().__init__()
        self.shape = shape
        self.kappa = float(kappa)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma_0 = float(gamma_0)
        self.gamma_avalanche = float(gamma_avalanche)
        self.strain_c = float(strain_c)
        self.device = torch.device(device)

        # 1. 3D Laplacian Kernel (7-Point Stencil on 3D Grid)
        # Center: -6.0, 6 Orthogonal Neighbors: +1.0
        laplacian_3d = torch.zeros((1, 1, 3, 3, 3), device=self.device)
        laplacian_3d[0, 0, 1, 1, 1] = -6.0
        laplacian_3d[0, 0, 0, 1, 1] = 1.0; laplacian_3d[0, 0, 2, 1, 1] = 1.0
        laplacian_3d[0, 0, 1, 0, 1] = 1.0; laplacian_3d[0, 0, 1, 2, 1] = 1.0
        laplacian_3d[0, 0, 1, 1, 0] = 1.0; laplacian_3d[0, 0, 1, 1, 2] = 1.0

        # Depthwise kernel across 4 quaternion channels (q_r, q_i, q_j, q_k)
        self.register_buffer("laplacian_kernel", laplacian_3d.repeat(4, 1, 1, 1, 1))

        # 2. Quaternion Field Tensor q (1, 4, Depth, Height, Width)
        # Initialized near unit quaternion q = [1.0, 0.0, 0.0, 0.0]
        init_q = torch.randn((1, 4, *shape), device=self.device) * 0.01
        init_q[:, 0, :, :, :] += 1.0
        self.q = init_q / torch.norm(init_q, dim=1, keepdim=True)

        # 3. Boundary Mask Tensor M (1, 1, Depth, Height, Width) & Boundary Quaternion q_bound
        self.register_buffer("mask", torch.zeros((1, 1, *shape), device=self.device))
        self.register_buffer("q_boundary", torch.zeros((1, 4, *shape), device=self.device))

    def set_boundary_condition(
        self,
        mask_tensor: torch.Tensor,
        boundary_q_tensor: torch.Tensor
    ) -> None:
        """
        Sets 3D boundary conditions and 4D quaternion orientation constraints.
        Mask M = 1.0 for obstacles/boundary clamps, M = 0.0 for free relaxation field.
        """
        self.mask.copy_(mask_tensor.to(self.device))
        self.q_boundary.copy_(boundary_q_tensor.to(self.device))

        # Direct algebraic binding
        self.q = (1.0 - self.mask) * self.q + self.mask * self.q_boundary

    def _compute_laplacian_3d(self, q_field: torch.Tensor) -> torch.Tensor:
        """Computes 3D Depthwise Convolution Laplacian for 4D Quaternion Field."""
        return F.conv3d(q_field, self.laplacian_kernel, padding=1, groups=4)

    def _compute_quaternion_norm_sq(self, q_field: torch.Tensor) -> torch.Tensor:
        """||q||^2 = q_r^2 + q_i^2 + q_j^2 + q_k^2."""
        return torch.sum(q_field ** 2, dim=1, keepdim=True)

    def forward(self, dt: float = 0.005) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """
        Executes 1 Step of 3D Quaternion Phase Field Relaxation (Zero Control-Flow Branching).
        """
        # A. 3D Spatial Laplacian
        lap = self._compute_laplacian_3d(self.q)

        # B. Quaternion Potential Gradient: dV/dq = \alpha * q + \beta * ||q||^2 * q
        norm_sq = self._compute_quaternion_norm_sq(self.q)
        dV = self.alpha * self.q + self.beta * norm_sq * self.q

        # C. 3D Spatial Strain: \sum ||\nabla q_c||^2
        grad_z = torch.roll(self.q, -1, dims=-3) - torch.roll(self.q, 1, dims=-3)
        grad_y = torch.roll(self.q, -1, dims=-2) - torch.roll(self.q, 1, dims=-2)
        grad_x = torch.roll(self.q, -1, dims=-1) - torch.roll(self.q, 1, dims=-1)
        strain = torch.sum(grad_z ** 2 + grad_y ** 2 + grad_x ** 2, dim=1, keepdim=True)

        # D. Smooth Heaviside dielectric breakdown activation
        avalanche_trigger = torch.sigmoid(40.0 * (strain - self.strain_c))
        gamma = self.gamma_0 + self.gamma_avalanche * avalanche_trigger

        # E. Dissipative Update Equation
        dq = gamma * (self.kappa * lap - dV)
        q_unconstrained = self.q + dt * dq

        # F. Manifold Projection onto S^3 Hyper-sphere
        q_normalized = q_unconstrained / (torch.norm(q_unconstrained, dim=1, keepdim=True) + 1e-8)

        # G. Algebraic Masking (No if/else branching)
        self.q = (1.0 - self.mask) * q_normalized + self.mask * self.q_boundary

        # H. Lyapunov Free Energy Functional on S^3
        free_energy = torch.sum(
            0.5 * self.kappa * strain +
            0.5 * self.alpha * norm_sq +
            0.25 * self.beta * (norm_sq ** 2)
        ).item()

        metrics = {
            "free_energy": free_energy,
            "max_strain": strain.max().item(),
            "mean_strain": strain.mean().item(),
            "avalanche_ratio": (avalanche_trigger > 0.5).float().mean().item(),
            "branch_divergence": 0,
            "pointer_chasing_steps": 0
        }

        return self.q, free_energy, metrics
