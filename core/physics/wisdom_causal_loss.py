"""
Wisdom-Causal Loss Engine (지혜-인과 손실 함수 엔진)

Formulates total system wisdom-causal loss:
  L_Wisdom-Causal = alpha * L_cascade + beta * L_macro-deform + gamma * L_entropy + delta * L_trinity

1. L_cascade: Scale cascade wave propagation energy across scale layers l in {1, ..., L}
2. L_macro-deform: Macro spacetime metric g_ij & 5D Clifford rotor R_macro deformation loss
3. L_entropy: Phase turbulence & information entropy measured by divergence of phase current density
   J_phase = sin(theta) * nabla(cos(theta)) - cos(theta) * nabla(sin(theta))
4. L_trinity: Trinitarian normalization & singularity avoidance for 1tan(theta) when cos(theta) -> 0
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math
import torch
import torch.nn as nn
import numpy as np


@dataclass
class WisdomLossComponents:
    l_wisdom_total: torch.Tensor
    l_cascade: torch.Tensor
    l_macro_deform: torch.Tensor
    l_entropy: torch.Tensor
    l_trinity: torch.Tensor
    max_turbulence: float
    trinity_regularity: float


class WisdomCausalLossEngine(nn.Module):
    """
    Computes L_Wisdom-Causal and executes multi-scale chain backpropagation.
    """

    def __init__(
        self,
        num_scales: int = 5,
        dimension: int = 64,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5,
        delta: float = 0.25,
        learning_rate_micro: float = 0.01,
        learning_rate_macro: float = 0.005,
        dtype=torch.float32
    ):
        super().__init__()
        self.num_scales = num_scales
        self.dimension = dimension
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lr_micro = learning_rate_micro
        self.lr_macro = learning_rate_macro
        self.dtype = dtype

        # Scale coupling weights w(l, l_0) across layers
        self.scale_weights = nn.Parameter(
            torch.tensor([1.0 / (1.0 + abs(l - 2)) for l in range(num_scales)], dtype=dtype)
        )

        # 5D Clifford rotor macro representation R_macro in Spin(5) (5x5 orthogonal matrix)
        init_rotor = torch.eye(5, dtype=dtype)
        self.r_macro = nn.Parameter(init_rotor)

        # Macro metric tensor g_ij
        self.g_macro = nn.Parameter(torch.eye(dimension, dtype=dtype))

    def compute_cascade_loss(
        self,
        micro_shock: torch.Tensor,
        scale_tensors: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculates L_cascade measuring total wave energy shift propagated across scale layers
        from a local micro-scale shock Delta theta_l0.
        """
        total_cascade = torch.tensor(0.0, dtype=self.dtype, device=micro_shock.device)

        # Compute shock magnitude
        shock_norm = torch.norm(micro_shock)

        for l in range(min(self.num_scales, len(scale_tensors))):
            w_l = scale_tensors[l] # [dimension] or [3, dimension]
            # Trinitarian wave tensor magnitude
            w_norm = torch.norm(w_l)
            w_coupling = torch.softmax(self.scale_weights, dim=0)[l]

            # Scale cascade energy transfer: w(l, l0) * ||W(x, lambda_l)||^2 * shock_norm
            layer_energy = w_coupling * (w_norm ** 2) * shock_norm
            total_cascade = total_cascade + layer_energy

        return total_cascade

    def compute_macro_deform_loss(
        self,
        r_macro_current: Optional[torch.Tensor] = None,
        g_macro_current: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates L_macro-deform evaluating deviation of 5D Clifford rotor R_macro
        and macro metric tensor g_ij from structural integrity (Identity state I_5 and I_dim).
        """
        if r_macro_current is None:
            r_macro_current = self.r_macro
        if g_macro_current is None:
            g_macro_current = self.g_macro

        # Deviation from Spin(5) identity matrix I_5
        identity_5 = torch.eye(5, dtype=self.dtype, device=r_macro_current.device)
        rotor_deform = torch.sum((r_macro_current - identity_5) ** 2)

        # Deviation of metric tensor from I_dim
        dim = g_macro_current.shape[0]
        identity_dim = torch.eye(dim, dtype=self.dtype, device=g_macro_current.device)
        metric_deform = torch.sum((g_macro_current - identity_dim) ** 2)

        return rotor_deform + 0.5 * metric_deform

    def compute_entropy_loss(
        self,
        phase_field: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Calculates L_entropy evaluating phase turbulence via phase current density:
        J_phase = sin(theta) * grad(cos(theta)) - cos(theta) * grad(sin(theta))
        L_entropy = integral (div(J_phase))^2 d x
        """
        sin_p = torch.sin(phase_field)
        cos_p = torch.cos(phase_field)

        # Spatial / sequential gradient approximation
        grad_cos = torch.gradient(cos_p, dim=-1)[0]
        grad_sin = torch.gradient(sin_p, dim=-1)[0]

        # Phase current density J_phase
        j_phase = sin_p * grad_cos - cos_p * grad_sin

        # Divergence of J_phase
        div_j = torch.gradient(j_phase, dim=-1)[0]

        # Turbulence penalty: sum( (div(J_phase))^2 )
        entropy_loss = torch.sum(div_j ** 2)
        max_turb = float(torch.max(torch.abs(div_j)).item())

        return entropy_loss, max_turb

    def compute_trinity_loss(
        self,
        phase_field: torch.Tensor,
        eps: float = 1e-4
    ) -> Tuple[torch.Tensor, float]:
        """
        Calculates L_trinity for 1tan(theta) singularity avoidance near cos(theta) -> 0
        and trinitarian geometric normalization: sin^2 + cos^2 = 1.
        """
        sin_p = torch.sin(phase_field)
        cos_p = torch.cos(phase_field)

        # Singularity avoidance: penalty when |cos_p| < eps
        cos_abs = torch.abs(cos_p)
        singularity_penalty = torch.sum(torch.relu(eps - cos_abs) ** 2)

        # Trinitarian normalization error: (sin^2 + cos^2 - 1)^2
        trinity_norm_err = torch.sum((sin_p ** 2 + cos_p ** 2 - 1.0) ** 2)

        # 1tan stability: 1tan = sin / (cos + sign(cos)*eps)
        safe_cos = torch.where(cos_abs < eps, eps * torch.sign(cos_p + 1e-8), cos_p)
        tan_p = sin_p / safe_cos
        tan_bound_penalty = torch.sum(torch.relu(torch.abs(tan_p) - 100.0) ** 2)

        l_trinity = trinity_norm_err + singularity_penalty + 0.01 * tan_bound_penalty
        regularity_val = float(1.0 / (1.0 + trinity_norm_err.item()))

        return l_trinity, regularity_val

    def forward(
        self,
        micro_shock: torch.Tensor,
        scale_tensors: List[torch.Tensor],
        phase_field: torch.Tensor,
        r_macro_current: Optional[torch.Tensor] = None,
        g_macro_current: Optional[torch.Tensor] = None
    ) -> WisdomLossComponents:
        """
        Calculates total Wisdom-Causal loss L_Wisdom-Causal.
        """
        l_cascade = self.compute_cascade_loss(micro_shock, scale_tensors)
        l_macro_deform = self.compute_macro_deform_loss(r_macro_current, g_macro_current)
        l_entropy, max_turb = self.compute_entropy_loss(phase_field)
        l_trinity, reg_val = self.compute_trinity_loss(phase_field)

        l_wisdom_total = (
            self.alpha * l_cascade +
            self.beta * l_macro_deform +
            self.gamma * l_entropy +
            self.delta * l_trinity
        )

        return WisdomLossComponents(
            l_wisdom_total=l_wisdom_total,
            l_cascade=l_cascade,
            l_macro_deform=l_macro_deform,
            l_entropy=l_entropy,
            l_trinity=l_trinity,
            max_turbulence=max_turb,
            trinity_regularity=reg_val
        )

    def execute_wisdom_backprop(
        self,
        loss_components: WisdomLossComponents,
        micro_params: List[nn.Parameter]
    ):
        """
        Executes scale-chain backpropagation from macro wisdom loss to micro parameters theta_l0.
        """
        # Zero gradients
        for p in micro_params:
            if p.grad is not None:
                p.grad.zero_()
        if self.r_macro.grad is not None:
            self.r_macro.grad.zero_()
        if self.g_macro.grad is not None:
            self.g_macro.grad.zero_()

        # Backward pass on Wisdom Loss
        loss_components.l_wisdom_total.backward(retain_graph=True)

        with torch.no_grad():
            # Update micro parameters
            for p in micro_params:
                if p.grad is not None:
                    p.data -= self.lr_micro * p.grad

            # Update macro rotor and metric tensor
            if self.r_macro.grad is not None:
                self.r_macro.data -= self.lr_macro * self.r_macro.grad
                # Project back to SO(5) via QR decomposition to preserve rotation geometry
                q, _ = torch.linalg.qr(self.r_macro.data)
                self.r_macro.data.copy_(q)

            if self.g_macro.grad is not None:
                self.g_macro.data -= self.lr_macro * self.g_macro.grad
                # Ensure symmetry
                self.g_macro.data.copy_(0.5 * (self.g_macro.data + self.g_macro.data.T))
