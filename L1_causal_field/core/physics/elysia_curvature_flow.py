r"""
Elysia Physics Subsystem: 3D Level-Set & Mean Curvature Flow Engine
===============================================================
Implements 3D Level-Set representation \Phi(\mathbf{x}, t) for phase boundary evolution,
Young-Laplace Mean Curvature Flow (MCF), and external potential wave advection.
Includes Narrow-Band spatial indexing to restrict computation to boundary regions |\Phi| <= \delta.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ElysiaCurvatureFlow3D(nn.Module):
    r"""
    elysia_engine: Level-Set Mean Curvature Flow & Young-Laplace Tensor Engine

    Solves the non-linear Level-Set PDE:
      \partial \Phi / \partial t = \gamma ||\nabla \Phi|| H + \langle \nabla V_{ext}, \nabla \Phi \rangle
    where H = 0.5 * div(\nabla \Phi / ||\nabla \Phi||) is the mean curvature.
    Supports optional Narrow-Band masking for sparse active voxel computation.
    """
    def __init__(
        self,
        gamma: float = 0.1,      # Surface tension / curvature flow speed
        dt: float = 0.01,        # Euler integration timestep
        delta: float = 0.5,      # Narrow-band boundary radius
        eps: float = 1e-6
    ):
        super().__init__()
        self.gamma = gamma
        self.dt = dt
        self.delta = delta
        self.eps = eps

        # 3D Central Difference kernels [1, 1, 3, 3, 3]
        dx = torch.zeros(1, 1, 3, 3, 3)
        dx[0, 0, 1, 1, 0], dx[0, 0, 1, 1, 2] = -0.5, 0.5

        dy = torch.zeros(1, 1, 3, 3, 3)
        dy[0, 0, 1, 0, 1], dy[0, 0, 1, 2, 1] = -0.5, 0.5

        dz = torch.zeros(1, 1, 3, 3, 3)
        dz[0, 0, 0, 1, 1], dz[0, 0, 2, 1, 1] = -0.5, 0.5

        self.register_buffer('kernel_dx', dx)
        self.register_buffer('kernel_dy', dy)
        self.register_buffer('kernel_dz', dz)

    def _gradient(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute 3D spatial gradient (gx, gy, gz) via 3D convolution."""
        gx = F.conv3d(field, self.kernel_dx, padding=1)
        gy = F.conv3d(field, self.kernel_dy, padding=1)
        gz = F.conv3d(field, self.kernel_dz, padding=1)
        return gx, gy, gz

    def compute_narrow_band_mask(self, phi: torch.Tensor) -> torch.Tensor:
        r"""Computes boolean active mask for narrow band |\Phi| <= \delta."""
        return torch.abs(phi) <= self.delta

    def forward(self, phi: torch.Tensor, v_ext: torch.Tensor = None, use_narrow_band: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            phi: [Batch, 1, D, H, W] current Phase Field (Level-Set)
            v_ext: [Batch, 1, D, H, W] external potential wave field
            use_narrow_band: If True, restrict updates to |\Phi| <= \delta
        Returns:
            tuple: (phi_next, H_mean_curvature)
        """
        # 1. First derivatives: Spatial gradient & norm
        gx, gy, gz = self._gradient(phi)
        grad_norm = torch.sqrt(gx.pow(2) + gy.pow(2) + gz.pow(2) + self.eps)

        # 2. Normalized normal vector field n = \nabla \Phi / ||\nabla \Phi||
        nx = gx / grad_norm
        ny = gy / grad_norm
        nz = gz / grad_norm

        # 3. Second derivatives: Divergence of normal vector field = 2 * Mean Curvature H
        dnx_dx, _, _ = self._gradient(nx)
        _, dny_dy, _ = self._gradient(ny)
        _, _, dnz_dz = self._gradient(nz)

        H = 0.5 * (dnx_dx + dny_dy + dnz_dz)

        # 4. Young-Laplace Surface Tension Flow (Curvature Flow towards Minimal Surface H=0)
        dphi_mcf = self.gamma * grad_norm * H

        # 5. External Potential Wave Coupling (Advection)
        if v_ext is not None:
            vx, vy, vz = self._gradient(v_ext)
            dphi_ext = -(vx * gx + vy * gy + vz * gz)
        else:
            dphi_ext = 0.0

        dphi_total = dphi_mcf + dphi_ext

        # 6. Apply Narrow-Band Active Masking
        if use_narrow_band:
            active_mask = self.compute_narrow_band_mask(phi)
            dphi_total = torch.where(active_mask, dphi_total, torch.zeros_like(dphi_total))

        # 7. Euler Time Step Integration
        phi_next = phi + self.dt * dphi_total

        return phi_next, H
