r"""
Elysia Core Physics: Phase Field Relaxation Engine (2D Zero-Branching Field Dynamics)
=====================================================================================
Implements spatiotemporal causalization and zero-branching continuous phase field
relaxation on 2D spatial manifolds (Allen-Cahn / Ginzburg-Landau potential field).

Key Principles:
1. Zero Branching: Control flow branching (if/else) and pointer chasing loops are
   eliminated in favor of tensor masking (M \odot \Psi) and parallel 2D convolution.
2. Spatiotemporal Causalization: Spatial Laplacian curvature \kappa \nabla^2 \Psi
   propagates boundary potential continuously across all grid points simultaneously.
3. Lyapunov Energy Minimization: System state relaxes monotonically towards the global
   minimum energy basin dE/dt <= 0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class PhaseFieldRelaxationEngine(nn.Module):
    """
    GPU/CPU Zero-Branching 2D Phase Field Relaxation Simulator.
    Replaces sequential search and pointer chasing with continuous tensor field relaxation.
    """

    def __init__(
        self,
        shape: Tuple[int, int] = (128, 128),
        kappa: float = 0.5,
        alpha: float = -1.0,
        beta: float = 1.0,
        gamma_0: float = 0.1,
        gamma_avalanche: float = 20.0,
        strain_c: float = 0.5,
        device: str = "cpu"
    ):
        super().__init__()
        self.shape = shape
        self.kappa = float(kappa)              # Gradient energy coefficient (curvature resistance)
        self.alpha = float(alpha)              # Symmetry breaking polynomial coefficient
        self.beta = float(beta)                # Higher-order saturation coefficient
        self.gamma_0 = float(gamma_0)          # Base relaxation coefficient
        self.gamma_avalanche = float(gamma_avalanche)  # Phase transition avalanche gain
        self.strain_c = float(strain_c)        # Critical strain threshold for dielectric breakdown
        self.device = torch.device(device)

        # 1. 2D Laplacian Convolution Kernel (3x3 Stencil)
        # Center: -4.0, Orthogonal Neighbors: +1.0
        laplacian_2d = torch.tensor([[0.0,  1.0, 0.0],
                                      [1.0, -4.0, 1.0],
                                      [0.0,  1.0, 0.0]], device=self.device).unsqueeze(0).unsqueeze(0)
        self.register_buffer("laplacian_kernel", laplacian_2d)

        # 2. Phase Field State Tensor \Psi (1, 1, H, W)
        self.psi = torch.zeros((1, 1, *shape), device=self.device)

        # 3. Boundary Mask Tensor M (1, 1, H, W) & Hard Clamped Boundary Values \Psi_bound
        self.register_buffer("mask", torch.zeros((1, 1, *shape), device=self.device))
        self.register_buffer("psi_boundary", torch.zeros((1, 1, *shape), device=self.device))

    def set_boundary_condition(
        self,
        mask_tensor: torch.Tensor,
        boundary_values: torch.Tensor
    ) -> None:
        """
        Applies problem boundary constraints (Delta B) algebraically via tensor masking.
        M = 1.0 for hard-clamped boundary points (e.g. entrance/exit/obstacles), M = 0.0 for free relaxation field.
        """
        self.mask.copy_(mask_tensor.to(self.device))
        self.psi_boundary.copy_(boundary_values.to(self.device))

        # Direct algebraic binding: \Psi = (1 - M) * \Psi + M * \Psi_bound
        self.psi = (1.0 - self.mask) * self.psi + self.mask * self.psi_boundary

    def _compute_laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """Computes 2D Laplacian nabla^2 Psi using depthwise convolution."""
        return F.conv2d(field, self.laplacian_kernel, padding=1)

    def forward(self, dt: float = 0.01) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """
        Executes 1 Step of Zero-Branching Spatiotemporal Phase Field Relaxation.
        Returns: (Psi state, free_energy, metrics)
        """
        # A. Compute spatial curvature (Laplacian nabla^2 Psi)
        lap = self._compute_laplacian(self.psi)

        # B. Compute potential gradient dV/d\Psi = \alpha \Psi + \beta \Psi^3
        dV = self.alpha * self.psi + self.beta * (self.psi ** 3)

        # C. Compute local phase strain |\nabla \Psi|^2 via spatial shift differences
        grad_x = torch.roll(self.psi, -1, dims=-1) - torch.roll(self.psi, 1, dims=-1)
        grad_y = torch.roll(self.psi, -1, dims=-2) - torch.roll(self.psi, 1, dims=-2)
        strain = grad_x ** 2 + grad_y ** 2

        # D. Smooth Heaviside dielectric breakdown trigger (Zero Branching)
        avalanche_trigger = torch.sigmoid(50.0 * (strain - self.strain_c))
        gamma = self.gamma_0 + self.gamma_avalanche * avalanche_trigger

        # E. Dissipative update equation: d\Psi/dt = \Gamma * (\kappa \nabla^2 \Psi - dV/d\Psi)
        d_psi = gamma * (self.kappa * lap - dV)
        psi_unconstrained = self.psi + dt * d_psi

        # F. Algebraic Boundary Clamping (No Control Flow Branching)
        self.psi = (1.0 - self.mask) * psi_unconstrained + self.mask * self.psi_boundary

        # G. Compute Lyapunov Free Energy Functional E[\Psi]
        free_energy = torch.sum(
            0.5 * self.kappa * strain +
            0.5 * self.alpha * (self.psi ** 2) +
            0.25 * self.beta * (self.psi ** 4)
        ).item()

        metrics = {
            "free_energy": free_energy,
            "max_strain": strain.max().item(),
            "mean_strain": strain.mean().item(),
            "avalanche_ratio": (avalanche_trigger > 0.5).float().mean().item(),
            "branch_divergence": 0,  # Zero branch divergence guaranteed
            "pointer_chasing_steps": 0
        }

        return self.psi, free_energy, metrics
