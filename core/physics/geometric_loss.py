r"""
Elysia Physics Subsystem: Geometric Loss Function
===============================================
Combines Yang-Mills Gauge Connection Curvature Loss, Kuramoto Phase-Lock Loss,
and Minimal Coupling Covariant Derivative Energy Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricLoss(nn.Module):
    r"""
    elysia_engine: Curvature Tensor & Phase-Lock Combined Geometric Loss Function

    L_geo = alpha * L_curvature + beta * L_phase + gamma * L_covariant
    """
    def __init__(self, alpha: float = 1.0, beta: float = 2.0, gamma: float = 0.5):
        super().__init__()
        self.alpha = alpha  # Curvature Loss Weight
        self.beta = beta    # Phase-Lock Loss Weight
        self.gamma = gamma  # Covariant Energy Weight

    def compute_curvature(self, omega: torch.Tensor) -> torch.Tensor:
        r"""
        omega: Gauge connection tensor of shape [Batch, Dim_SpaceTime, Dim_State, Dim_State]
               (where Dim_SpaceTime >= 2 for temporal and spatial connection components)
        Computes F_{\mu\nu} = \partial_\mu \Omega_\nu - \partial_\nu \Omega_\mu + [\Omega_\mu, \Omega_\nu]
        """
        if omega.dim() < 4 or omega.shape[1] < 2:
            # Fallback for simplified connection tensor [Batch, Dim_State, Dim_State]
            return torch.norm(omega, p='fro', dim=(-2, -1)).pow(2).mean()

        omega_t = omega[:, 0]
        omega_x = omega[:, 1]

        # Commutator [\Omega_\mu, \Omega_\nu] = \Omega_\mu \Omega_\nu - \Omega_\nu \Omega_\mu
        commutator = torch.matmul(omega_t, omega_x) - torch.matmul(omega_x, omega_t)

        d_omega_x = torch.diff(omega, dim=1, prepend=omega[:, :1])
        d_omega = d_omega_x[:, 1] - d_omega_x[:, 0]

        F_mu_nu = d_omega + commutator
        return torch.norm(F_mu_nu, p='fro', dim=(-2, -1)).pow(2).mean()

    def compute_phase_loss(self, z_low: torch.Tensor, z_fb: torch.Tensor) -> torch.Tensor:
        """
        z_low, z_fb: Complex Tensors [Batch, N]
        """
        phase_low = torch.angle(z_low)
        phase_fb = torch.angle(z_fb)
        delta_phase = phase_low - phase_fb

        # Order Parameter S_lock
        cos_mean = torch.mean(torch.cos(delta_phase), dim=-1)
        sin_mean = torch.mean(torch.sin(delta_phase), dim=-1)
        S_lock = torch.sqrt(cos_mean.pow(2) + sin_mean.pow(2) + 1e-8)

        return (1.0 - S_lock).mean()

    def compute_covariant_loss(self, psi: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        r"""
        psi: State Tensor [Batch, Dim_State] or [Batch, ..., Dim_State]
        omega: Connection Tensor [Batch, Dim_SpaceTime, Dim_State, Dim_State]
        """
        # Grad of Psi (simplified spatial/temporal difference)
        d_psi = torch.diff(psi, dim=-1, prepend=psi[..., :1])

        # Covariant Derivative: D_\mu \Psi = \partial_\mu \Psi + \Omega_\mu \Psi
        if omega.dim() == 4:
            conn = omega[:, 0]
        else:
            conn = omega

        if psi.dim() == 2:
            omega_psi = torch.matmul(conn, psi.unsqueeze(-1)).squeeze(-1)
        else:
            omega_psi = torch.matmul(conn, psi.unsqueeze(-1)).squeeze(-1)

        D_psi = d_psi + omega_psi

        return torch.norm(D_psi, p=2, dim=-1).pow(2).mean()

    def forward(
        self,
        psi: torch.Tensor,
        z_low: torch.Tensor,
        z_fb: torch.Tensor,
        omega: torch.Tensor
    ) -> dict:

        l_curv = self.compute_curvature(omega)
        l_phase = self.compute_phase_loss(z_low, z_fb)
        l_cov = self.compute_covariant_loss(psi, omega)

        total_loss = (self.alpha * l_curv) + (self.beta * l_phase) + (self.gamma * l_cov)

        return {
            "total_loss": total_loss,
            "l_curvature": l_curv.item(),
            "l_phase_lock": l_phase.item(),
            "l_covariant": l_cov.item()
        }
