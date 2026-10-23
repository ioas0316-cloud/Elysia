"""
Multi-Scale Phase Loss Engine & Dual Backpropagation System (다층 위상 장 손실 함수 및 이중 역전파)

Formulates total system loss:
  L_total = alpha * L_scale + beta * L_causal + gamma * L_friction

1. L_scale: Inter-scale Phase Lock Loss across scale parameter s in [0, 1]
2. L_causal: Asymmetric Causal Temporal Entropy & Metric Tensor Loss (g_ij != g_ji)
3. L_friction: Exogenous Reality Deformation Strain Loss
4. Dual Backpropagation: Simultaneous update of weights Theta and Spacetime Metric g_ij (Ricci-flow deformation)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import numpy as np


@dataclass
class LossComponents:
    l_total: torch.Tensor
    l_scale: torch.Tensor
    l_causal: torch.Tensor
    l_friction: torch.Tensor
    is_critical_friction: bool
    scale_phase_diff: float
    entropy_rate: float


class MultiScalePhaseLossEngine(nn.Module):
    """
    Computes L_total = alpha * L_scale + beta * L_causal + gamma * L_friction,
    and executes Dual Backpropagation on weights Theta and Spacetime Metric g_ij.
    """

    def __init__(
        self,
        dimension: int = 64,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.5,
        critical_friction_threshold: float = 2.5,
        learning_rate_theta: float = 0.01,
        learning_rate_metric: float = 0.005,
        dtype=torch.float32
    ):
        super().__init__()
        self.dimension = dimension
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.critical_friction_threshold = critical_friction_threshold
        self.lr_theta = learning_rate_theta
        self.lr_g = learning_rate_metric
        self.dtype = dtype

        # Asymmetric Metric Tensor g_ij where g_ij != g_ji to enforce temporal arrow
        # Initialized near identity + slight asymmetric skew
        init_g = torch.eye(dimension, dtype=dtype) + 0.05 * torch.randn(dimension, dimension, dtype=dtype)
        self.metric_tensor_g = nn.Parameter(init_g)

        # Stiffness tensor K_stiffness for strain energy evaluation
        self.stiffness_matrix = nn.Parameter(torch.eye(dimension, dtype=dtype) * 2.0)

        # Renormalization Operator R_RG for multi-scale coarse graining
        self.rg_operator = nn.Linear(dimension, dimension, bias=False, dtype=dtype)

    def compute_scale_loss(
        self,
        psi_micro: torch.Tensor,
        phi_macro: torch.Tensor,
        num_scale_steps: int = 5
    ) -> Tuple[torch.Tensor, float]:
        """
        Computes L_scale evaluating inter-scale alignment between micro token field (s=0)
        and macro intent field (s=1).
        """
        # Interpolate field across scale manifold s in [0, 1]
        scale_field = []
        for i in range(num_scale_steps):
            s = i / max(1, num_scale_steps - 1)
            # Linear/harmonic blend between micro and macro
            psi_s = (1.0 - s) * psi_micro + s * phi_macro
            scale_field.append(psi_s)

        # Compute gradient wrt scale parameter s: nabla_s Psi
        scale_grad_loss = torch.tensor(0.0, dtype=self.dtype, device=psi_micro.device)
        for i in range(num_scale_steps - 1):
            d_psi_ds = scale_field[i + 1] - scale_field[i]
            rg_psi = self.rg_operator(scale_field[i])
            scale_grad_loss = scale_grad_loss + torch.sum((d_psi_ds - rg_psi) ** 2)

        # Phase locking term: 1 - cos(phi_macro - phi_micro)
        phase_micro = torch.atan2(psi_micro, torch.roll(psi_micro, shifts=1) + 1e-8).mean()
        phase_macro = torch.atan2(phi_macro, torch.roll(phi_macro, shifts=1) + 1e-8).mean()
        phase_lock_term = 1.0 - torch.cos(phase_macro - phase_micro)

        l_scale = scale_grad_loss + phase_lock_term
        phase_diff_val = float(torch.abs(phase_macro - phase_micro).item())

        return l_scale, phase_diff_val

    def compute_causal_loss(
        self,
        psi_curr: torch.Tensor,
        psi_prev: torch.Tensor,
        f_causal_expected: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Computes L_causal enforcing asymmetric temporal metric velocity and entropy rate penalty.
        """
        # Velocity dot_Psi = Psi(t) - Psi(t - dt)
        dot_psi = psi_curr - psi_prev # [dimension]

        # Metric velocity quadratic form: dot_psi^i * g_ij * dot_psi^j
        metric_velocity = torch.matmul(dot_psi.unsqueeze(0), torch.matmul(self.metric_tensor_g, dot_psi.unsqueeze(1))).squeeze()

        # Discrepancy with internal causal dynamics F_causal
        causal_field_diff = torch.sum((metric_velocity - f_causal_expected) ** 2)

        # Entropy estimation: S(Psi) = - sum(p * log(p))
        p_curr = torch.softmax(torch.abs(psi_curr), dim=-1)
        p_prev = torch.softmax(torch.abs(psi_prev), dim=-1)

        s_curr = -torch.sum(p_curr * torch.log(p_curr + 1e-8))
        s_prev = -torch.sum(p_prev * torch.log(p_prev + 1e-8))

        # Entropy rate dS/dt
        ds_dt = s_curr - s_prev
        # Penalty if entropy rate is negative (unnatural retroactive spontaneous decay)
        entropy_penalty = torch.relu(-ds_dt)

        l_causal = entropy_penalty + causal_field_diff
        return l_causal, float(ds_dt.item())

    def compute_friction_loss(
        self,
        f_causal_pred: torch.Tensor,
        omega_ext: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes L_friction strain energy from exogenous reality shock:
        L_friction = 0.5 * Tr((F_causal - Omega_ext)^T * K_stiffness * (F_causal - Omega_ext))
        """
        diff = (f_causal_pred - omega_ext).unsqueeze(1) # [dimension, 1]
        strain_energy = 0.5 * torch.matmul(diff.T, torch.matmul(self.stiffness_matrix, diff)).squeeze()
        return strain_energy

    def forward(
        self,
        psi_micro: torch.Tensor,
        phi_macro: torch.Tensor,
        psi_prev: torch.Tensor,
        f_causal_pred: torch.Tensor,
        omega_ext: torch.Tensor
    ) -> LossComponents:
        """
        Calculates total loss L_total and checks for critical friction status.
        """
        l_scale, phase_diff = self.compute_scale_loss(psi_micro, phi_macro)
        l_causal, entropy_rate = self.compute_causal_loss(psi_micro, psi_prev, f_causal_pred)
        l_friction = self.compute_friction_loss(f_causal_pred, omega_ext)

        l_total = self.alpha * l_scale + self.beta * l_causal + self.gamma * l_friction

        is_critical = float(l_friction.item()) > self.critical_friction_threshold

        return LossComponents(
            l_total=l_total,
            l_scale=l_scale,
            l_causal=l_causal,
            l_friction=l_friction,
            is_critical_friction=is_critical,
            scale_phase_diff=phase_diff,
            entropy_rate=entropy_rate
        )

    def execute_dual_backprop(
        self,
        loss_components: LossComponents,
        parameters_theta: List[nn.Parameter]
    ):
        """
        Executes Dual Backpropagation:
        1. Standard gradient update on model weights Theta: dTheta/dt = -lr * grad_Theta(L_total)
        2. Metric Deformation (Ricci Flow style) on Spacetime Metric g_ij: dg_ij/dt = -lr_g * grad_g(L_friction)
        """
        # Zero prior grads
        for p in parameters_theta:
            if p.grad is not None:
                p.grad.zero_()
        if self.metric_tensor_g.grad is not None:
            self.metric_tensor_g.grad.zero_()

        # Backward pass on total loss for Theta
        loss_components.l_total.backward(retain_graph=True)

        with torch.no_grad():
            for p in parameters_theta:
                if p.grad is not None:
                    p.data -= self.lr_theta * p.grad

            # Metric deformation if friction is critical
            if loss_components.is_critical_friction:
                # Ricci flow metric deformation using L_friction gradient
                if self.metric_tensor_g.grad is not None:
                    self.metric_tensor_g.data -= self.lr_g * self.metric_tensor_g.grad
                    # Enforce asymmetric metric positivity / non-degeneracy
                    self.metric_tensor_g.data += 0.01 * torch.eye(self.dimension, dtype=self.dtype)
