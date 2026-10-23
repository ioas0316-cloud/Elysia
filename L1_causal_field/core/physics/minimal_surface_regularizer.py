r"""
Elysia Physics Subsystem: Minimal Surface & Young-Laplace Equilibrium Regularizer
================================================================================
Prevents phase collapse and absolute zero freezing (overfitting) by balancing
surface tension contraction (\int \sqrt{\det(g)}) and internal information volume pressure,
supplemented with effective thermal zero-point fluctuations (T_eff).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimalSurfaceRegularizer(nn.Module):
    r"""
    elysia_engine: Minimal Surface & Young-Laplace Equilibrium Regularizer

    Balances Surface Tension (\gamma) vs. Internal Volume Pressure (\rho) & Phase Entropy (\sigma).
    Injects effective zero-point thermal fluctuation (T_eff) to avoid phase freezing.
    """
    def __init__(
        self,
        gamma_tension: float = 1e-3,   # Surface tension coefficient
        rho_pressure: float = 5e-4,    # Volume pressure coefficient
        sigma_entropy: float = 1e-3,   # Phase entropy coefficient
        t_effective: float = 1e-4,     # Thermal fluctuation temperature
        eps: float = 1e-8
    ):
        super().__init__()
        self.gamma = gamma_tension
        self.rho = rho_pressure
        self.sigma = sigma_entropy
        self.t_eff = t_effective
        self.eps = eps

    def compute_induced_metric(self, psi: torch.Tensor) -> torch.Tensor:
        r"""
        psi: [Batch, N_points, Dim_State] or [Batch, Dim_State]
        Computes determinant of induced metric g_{\mu\nu} via discrete spatial differences.
        """
        if psi.dim() == 2:
            psi = psi.unsqueeze(1)  # [Batch, 1, Dim_State]

        d_psi_x = torch.diff(psi, dim=1, prepend=psi[:, :1, :])
        if psi.dim() > 3:
            d_psi_y = torch.diff(psi, dim=2, prepend=psi[:, :, :1])
        else:
            d_psi_y = d_psi_x

        g11 = torch.sum(d_psi_x * d_psi_x, dim=-1)
        g22 = torch.sum(d_psi_y * d_psi_y, dim=-1)
        g12 = torch.sum(d_psi_x * d_psi_y, dim=-1)

        det_g = torch.clamp(g11 * g22 - g12.pow(2), min=self.eps)
        return det_g

    def compute_phase_entropy(self, psi_complex: torch.Tensor) -> torch.Tensor:
        """
        Computes differential phase entropy of complex state tensor field.
        """
        angles = torch.angle(psi_complex)
        phase_var = torch.var(angles, dim=-1)
        phase_entropy = 0.5 * torch.log(2.0 * torch.pi * torch.e * phase_var + self.eps)
        return phase_entropy.mean()

    def forward(self, psi_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            psi_state: Complex or real tensor field state
        Returns:
            torch.Tensor: Equilibrium regularization loss
        """
        if not psi_state.is_complex():
            half_dim = psi_state.shape[-1] // 2
            psi_complex = torch.complex(psi_state[..., :half_dim], psi_state[..., half_dim:])
        else:
            psi_complex = psi_state

        det_g = self.compute_induced_metric(psi_state.real if psi_state.is_complex() else psi_state)

        # 1. Surface Tension Loss (\int \sqrt{\det(g)})
        surface_area = torch.sqrt(det_g).mean()
        loss_surface = self.gamma * surface_area

        # 2. Internal Pressure Loss (Volume expansion & Phase entropy)
        volume_expansion = -self.rho * torch.log(det_g + self.eps).mean()
        phase_entropy = -self.sigma * self.compute_phase_entropy(psi_complex)
        loss_pressure = volume_expansion + phase_entropy

        total_reg_loss = loss_surface + loss_pressure

        # 3. Thermal fluctuation injection during training
        if self.training and self.t_eff > 0:
            noise = torch.randn_like(psi_state.real if psi_state.is_complex() else psi_state) * torch.sqrt(torch.tensor(2.0 * self.t_eff))
            if psi_state.is_complex():
                psi_state.data.add_(torch.complex(noise, torch.zeros_like(noise)))
            else:
                psi_state.data.add_(noise)

        return total_reg_loss
