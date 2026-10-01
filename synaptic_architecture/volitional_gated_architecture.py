"""
Elysia PyTorch 4-Layer Phase-Gated Volitional Architecture & Multi-Agent Kuramoto Field Engine
===================================================================================================
Implements:
1. VolitionalGatedArchitecturePyTorch (torch.nn.Module):
   - Layer 0: Immutable Substrate (C_max capacity, tau_min dissonance threshold, non-differentiable environmental metrics).
   - Layer 1: Passive Resonance Gate (Warp-level zero-cost early exit check when error < tau_min).
   - Layer 2: Volitional Routing Gate (nn.Parameter attention vector v_attn & orientation tensor T_orient).
   - Layer 3: Local Phase Engine (Lie Group SO(N) / Lie Algebra so(N) exponential map manifold refolding).

2. DistributedKuramotoPhaseField (torch.nn.Module):
   - Multi-agent local agency fields A_k each with individual attention & orientation tensors.
   - Dynamic coupling strength K(E) = K_0 / (1 + exp(-(E - tau_min) / delta)).
   - Kuramoto phase locking order parameter R(t) transition:
     * E < tau_min -> R ~ 0 (Decoherent multi-view curiosity exploration).
     * E >= tau_min -> R -> 1 (Phase-locked single-minded collective intent).
   - Manifold-preserving SO(N) Lie exponential field update.
"""

import math
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def matrix_exponential_so_n(omega: torch.Tensor) -> torch.Tensor:
    """
    Computes matrix exponential exp(Omega) for a skew-symmetric Lie Algebra tensor Omega in so(N).
    Ensures orthogonal rotation matrix exp(Omega) in SO(N) where R @ R.T = I.
    Supports 2D (N, N) and 3D (B, N, N) tensors.
    """
    # Guarantee skew-symmetry
    omega_skew = 0.5 * (omega - omega.transpose(-1, -2))
    return torch.linalg.matrix_exp(omega_skew)


class VolitionalGatedArchitecturePyTorch(nn.Module):
    """
    PyTorch implementation of the 4-Layer Phase-Gated Architecture.
    Coexistence of environmental immutable constraints and volitional subject agency.
    """
    def __init__(self, topology_dim: int = 16, c_max: float = 0.35, tau_min: float = 0.05):
        super().__init__()
        self.topology_dim = topology_dim

        # [Layer 0: Immutable Substrate] Absolute environmental constraints (Non-trainable, non-modifiable)
        self.register_buffer("c_max", torch.tensor(c_max, dtype=torch.float32))
        self.register_buffer("tau_min", torch.tensor(tau_min, dtype=torch.float32))

        # [Layer 2: Volitional Agency] Free parameters adjustable by the subject/agent
        self.volitional_attention = nn.Parameter(torch.randn(topology_dim) * 0.1)
        self.orientation_tensor = nn.Parameter(torch.randn(topology_dim, topology_dim) * 0.1)

    def forward(
        self,
        x_input: torch.Tensor,
        internal_state: torch.Tensor
    ) -> Dict[str, Any]:
        """
        x_input: (Topology_Dim,) or (Batch, Topology_Dim)
        internal_state: (Topology_Dim,) or (Batch, Topology_Dim)
        """
        # 1. [Layer 0] Prediction Error Residual E = ||X - Phi||_2
        diff = x_input - internal_state
        prediction_error = torch.norm(diff, p=2, dim=-1)

        mean_error = prediction_error.mean().item()

        # 2. [Layer 1] Passive Resonance Gate Check (Automated early exit)
        if mean_error < self.tau_min.item():
            return {
                "new_internal_state": internal_state.clone(),
                "compute_cost": 0.0,
                "prediction_error": mean_error,
                "passive_gated": True,
                "volitional_active": False,
                "orthogonality_error": 0.0
            }

        # 3. [Layer 2] Volitional Routing Gate Activation
        # Attention focus weighted by error pressure
        attn_weighted = self.volitional_attention * prediction_error.unsqueeze(-1) if prediction_error.dim() > 0 else self.volitional_attention * prediction_error
        selected_focus = F.softmax(attn_weighted, dim=-1)

        directional_intent = torch.matmul(selected_focus, self.orientation_tensor.T)

        # 4. [Layer 3] Local Phase Engine Execution (Clamped by environmental C_max capacity)
        bounded_compute = torch.clamp(directional_intent, -self.c_max, self.c_max)

        # Lie Algebra skew-symmetric generator Omega from difference vector (bivector wedge outer product)
        if diff.dim() == 1:
            diff_col = diff.unsqueeze(1)
            diff_row = diff.unsqueeze(0)
            omega_raw = torch.matmul(diff_col, diff_row) - torch.matmul(diff_row.T, diff_col.T)
        else:
            omega_raw = torch.bmm(diff.unsqueeze(2), diff.unsqueeze(1)) - torch.bmm(diff.unsqueeze(1), diff.unsqueeze(2))

        eta_E = 0.05 * (mean_error - self.tau_min.item())
        g_exp = matrix_exponential_so_n(eta_E * omega_raw)

        # Check orthogonality preservation ||g^T g - I||
        if g_exp.dim() == 2:
            ortho_err = torch.norm(torch.matmul(g_exp.T, g_exp) - torch.eye(self.topology_dim, device=x_input.device)).item()
            new_internal_state = torch.matmul(internal_state, g_exp) + bounded_compute
        else:
            ortho_err = torch.norm(torch.bmm(g_exp.transpose(1, 2), g_exp) - torch.eye(self.topology_dim, device=x_input.device)).item()
            new_internal_state = torch.bmm(internal_state.unsqueeze(1), g_exp).squeeze(1) + bounded_compute

        return {
            "new_internal_state": new_internal_state,
            "compute_cost": torch.norm(bounded_compute).item(),
            "prediction_error": mean_error,
            "passive_gated": False,
            "volitional_active": True,
            "selected_focus": selected_focus,
            "directional_intent": directional_intent,
            "g_exp": g_exp,
            "orthogonality_error": ortho_err
        }


class DistributedKuramotoPhaseField(nn.Module):
    """
    Distributed Tensor Field with N local agency fields and Kuramoto Phase Locking.
    Models multi-agent Phase Locking order parameter transition R(t) driven by dissonance error E:
    - E < tau_min: Coupling K(E) -> 0, R -> 0 (Decoherent multi-view exploration).
    - E >= tau_min: Coupling K(E) >> K_c, R -> 1 (Single-minded collective intent).
    """
    def __init__(
        self,
        num_agents: int = 8,
        topology_dim: int = 16,
        tau_min: float = 0.10,
        c_max: float = 0.50,
        k_0: float = 5.0,
        delta: float = 0.02
    ):
        super().__init__()
        self.num_agents = num_agents
        self.topology_dim = topology_dim
        self.tau_min = tau_min
        self.c_max = c_max
        self.k_0 = k_0
        self.delta = delta

        # Local agency parameters for each agent
        self.agent_attentions = nn.Parameter(torch.randn(num_agents, topology_dim) * 0.1)
        self.agent_orientations = nn.Parameter(torch.randn(num_agents, topology_dim, topology_dim) * 0.1)

        # Agent natural frequencies omega_k and initial phases theta_k
        self.register_buffer("natural_frequencies", torch.randn(num_agents) * 0.5)
        self.phases = nn.Parameter(torch.rand(num_agents) * 2.0 * math.pi)

    def compute_coupling_strength(self, E: float) -> float:
        """K(E) = K_0 / (1 + exp(-(E - tau_min) / delta))"""
        sig = 1.0 / (1.0 + math.exp(-(E - self.tau_min) / self.delta))
        return float(self.k_0 * sig)

    def compute_order_parameter(self) -> Tuple[float, float]:
        """
        Kuramoto Order Parameter: Z = R * exp(i * psi) = (1/N) * sum_k exp(i * theta_k)
        Returns: (R, psi)
        """
        cos_sum = torch.sum(torch.cos(self.phases)).item()
        sin_sum = torch.sum(torch.sin(self.phases)).item()

        R = math.sqrt(cos_sum**2 + sin_sum**2) / self.num_agents
        psi = math.atan2(sin_sum, cos_sum)
        return R, psi

    def step(
        self,
        x_input: torch.Tensor,
        internal_state: torch.Tensor,
        dt: float = 0.01
    ) -> Dict[str, Any]:
        """
        x_input: (Topology_Dim,)
        internal_state: (Topology_Dim,)
        """
        # 1. Residual calculation
        diff = x_input - internal_state
        prediction_error = torch.norm(diff, p=2).item()

        # 2. Dynamic coupling strength K(E)
        K_E = self.compute_coupling_strength(prediction_error)

        # 3. Kuramoto phase dynamics update: dTheta_i/dt = omega_i + (K(E)/N) * sum_j sin(theta_j - theta_i)
        d_theta = self.natural_frequencies.clone()
        if K_E > 1e-5:
            phase_diffs = self.phases.unsqueeze(0) - self.phases.unsqueeze(1) # theta_j - theta_i
            coupling_term = (K_E / self.num_agents) * torch.sum(torch.sin(phase_diffs), dim=1)
            d_theta = d_theta + coupling_term

        # Euler step for phases
        with torch.no_grad():
            self.phases.add_(d_theta * dt)
            self.phases.remainder_(2.0 * math.pi)

        # Order parameter R and mean phase psi
        R, psi = self.compute_order_parameter()

        # 4. Agent vector emission and interference aggregation
        agent_intents = []
        for k in range(self.num_agents):
            focus = F.softmax(self.agent_attentions[k] * prediction_error, dim=-1)
            intent = torch.matmul(self.agent_orientations[k], focus)
            # Modulate intent by agent phase alignment
            phase_weight = math.cos(self.phases[k].item() - psi)
            agent_intents.append(intent * phase_weight)

        collective_intent = torch.stack(agent_intents, dim=0).sum(dim=0)

        # Bounded compute by C_max
        bounded_intent = torch.clamp(collective_intent, -self.c_max, self.c_max)

        # Lie Group SO(N) collective transformation
        if prediction_error >= self.tau_min:
            diff_col = diff.unsqueeze(1)
            diff_row = diff.unsqueeze(0)
            omega_raw = torch.matmul(diff_col, diff_row) - torch.matmul(diff_row.T, diff_col.T)
            g_exp = matrix_exponential_so_n(0.05 * R * omega_raw)
            updated_state = torch.matmul(internal_state, g_exp) + bounded_intent
            ortho_err = torch.norm(torch.matmul(g_exp.T, g_exp) - torch.eye(self.topology_dim, device=x_input.device)).item()
        else:
            updated_state = internal_state.clone()
            g_exp = torch.eye(self.topology_dim, device=x_input.device)
            ortho_err = 0.0

        return {
            "prediction_error": prediction_error,
            "coupling_strength": K_E,
            "order_parameter_R": R,
            "mean_phase_psi": psi,
            "collective_intent": bounded_intent,
            "new_internal_state": updated_state,
            "orthogonality_error": ortho_err,
            "is_phase_locked": R > 0.8
        }
