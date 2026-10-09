"""
Dual-Frame Causal Engine & Theory of Mind Architecture for Elysia.

This module implements the 2nd-person ToM and scale-hierarchy dynamics:
1. TheoryOfMindObserver: Reconstructs speaker/other's internal logic metric G^(other)
   and Clifford frame rotor R_ToM via action minimization argmin_G δS_other.
2. CliffordMultivectorMemory: Multivector memory grid over Cl_3 algebra (8 grade basis elements)
   supporting orthogonal grade projection (Grade 0: Scalar/Actual, Grade 1: Vector/1st-order CF,
   Grade 2: Bivector/2nd-order CF Rotors, Grade 3: Pseudoscalar/Contextual Volume).
3. CounterfactualWaveEngine: Handles causal advection, time-reversed rewind, e^(iπ) phase inversion
   destructive interference, and counterfactual re-projection.
4. KuramotoDualFrameCoupler: Dual-frame phase coupling, 180° (π rad) topological deadlock detection
   and orthogonal bivector rotor unlocking, Lyapunov energy V(t) decay, and phase locking R -> 1.0.
5. ScaleRenormalizationEngine: 4D scale-space (x, y, z, s) dynamics with Wilsonian scale coarse-graining,
   Clifford bivector grade promotion (v_i ∧ v_j -> B), and HJB back-projection top-down constraint torque.
"""

import math
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn


class TheoryOfMindObserver(nn.Module):
    """
    2nd-Person Theory of Mind Observer.
    Inverts the speaker/other's internal logic metric G^(other) and Clifford frame rotor R_ToM
    from observed statement wave trajectories using action minimization: argmin_G δS_other.
    """

    def __init__(self, dim: int = 4, device: Optional[str] = None):
        super().__init__()
        self.dim = dim
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dev = torch.device(self.device_str)

        self.register_buffer("G_self", torch.eye(dim, dtype=torch.float32, device=dev))

    def reconstruct_other_metric(
        self,
        psi_statement: torch.Tensor,
        lr: float = 0.05,
        steps: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Reconstructs G^(other) and Clifford rotor R_ToM that minimizes geodesic action S_other
        for the given statement wave psi_statement.
        Returns: (G_other, R_tom, min_action)
        """
        dev = psi_statement.device
        # Initialize trainable matrix parameter for G_other
        G_param = torch.eye(self.dim, dtype=torch.float32, device=dev, requires_grad=True)
        rotor_angle = torch.zeros(1, dtype=torch.float32, device=dev, requires_grad=True)

        optimizer = torch.optim.Adam([G_param, rotor_angle], lr=lr)

        psi_flat = psi_statement.reshape(-1, self.dim) if psi_statement.dim() > 2 else psi_statement

        min_action_val = 0.0
        for step in range(steps):
            optimizer.zero_grad()

            # Ensure G_other is symmetric positive definite
            G_other = 0.5 * (G_param + G_param.t()) + 0.1 * torch.eye(self.dim, device=dev)

            # Geodesic action S = 1/2 * mean(psi @ G_other @ psi^T)
            # Statement flows smoothly along geodesics in G_other frame
            G_inv = torch.linalg.inv(G_other)
            kinetic_energy = torch.sum(psi_flat * torch.matmul(psi_flat, G_inv), dim=-1)
            action_S = 0.5 * kinetic_energy.mean()

            # Add metric regularization (distance from baseline identity)
            reg = 0.01 * torch.norm(G_other - torch.eye(self.dim, device=dev), p="fro")
            total_loss = action_S + reg

            total_loss.backward()
            optimizer.step()

            min_action_val = total_loss.item()

        with torch.no_grad():
            G_other_final = 0.5 * (G_param + G_param.t()) + 0.1 * torch.eye(self.dim, device=dev)
            theta = rotor_angle.item()
            # 2D/4D Clifford bivector rotor representation
            cos_t = math.cos(theta / 2.0)
            sin_t = math.sin(theta / 2.0)
            R_tom = torch.eye(self.dim, device=dev)
            if self.dim >= 2:
                R_tom[0, 0] = cos_t
                R_tom[0, 1] = -sin_t
                R_tom[1, 0] = sin_t
                R_tom[1, 1] = cos_t

        return G_other_final.detach(), R_tom.detach(), min_action_val


class CliffordMultivectorMemory:
    """
    3D Clifford Geometric Algebra Cl_3 Multivector Memory Tensor.
    Grades 0-3 over 8 basis elements:
    [Scalar(1), V_x(e1), V_y(e2), V_z(e3), B_xy(e12), B_yz(e23), B_zx(e31), T_xyz(e123)]
    Supports orthogonal grade projection without cross-contamination.
    """

    def __init__(self, depth: int = 16, height: int = 16, width: int = 16, device: Optional[str] = None):
        self.shape = (depth, height, width, 8)
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dev = torch.device(self.device_str)
        self.memory_field = torch.zeros(self.shape, dtype=torch.float32, device=dev)

    def write_scenario(self, grade: int, scenario_tensor: torch.Tensor):
        """
        Writes counterfactual wave scenario into designated Clifford grade channel (0..3).
        """
        dev = self.memory_field.device
        s_tensor = scenario_tensor.to(dev)

        if grade == 0:     # Grade 0: Scalar (Actual consensus history)
            self.memory_field[..., 0:1] += s_tensor if s_tensor.dim() == 4 else s_tensor.unsqueeze(-1)
        elif grade == 1:   # Grade 1: Vector (Vx, Vy, Vz - 1st order CF displacement)
            self.memory_field[..., 1:4] += s_tensor
        elif grade == 2:   # Grade 2: Bivector (Bxy, Byz, Bzx - 2nd order CF rotors)
            self.memory_field[..., 4:7] += s_tensor
        elif grade == 3:   # Grade 3: Pseudoscalar (Txyz - 3rd order CF volume inversion)
            self.memory_field[..., 7:8] += s_tensor if s_tensor.dim() == 4 else s_tensor.unsqueeze(-1)
        else:
            raise ValueError(f"Invalid Clifford Grade {grade}. Must be 0, 1, 2, or 3.")

    def read_grade_projection(self, grade: int) -> torch.Tensor:
        """
        Extracts orthogonal grade projection <M>_k without interference.
        """
        if grade == 0:
            return self.memory_field[..., 0:1]
        elif grade == 1:
            return self.memory_field[..., 1:4]
        elif grade == 2:
            return self.memory_field[..., 4:7]
        elif grade == 3:
            return self.memory_field[..., 7:8]
        else:
            raise ValueError(f"Invalid Clifford Grade {grade}. Must be 0, 1, 2, or 3.")


class CounterfactualWaveEngine:
    """
    Counterfactual Wave Reasoning & Intervention Engine.
    Handles causal advection, time-reversed rewind, e^(iπ) phase inversion (-1 factor)
    destructive interference event cancellation, and counterfactual re-forward projection.
    """

    def __init__(self, device: Optional[str] = None):
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def compute_counterfactual_wave_branch(
        self,
        psi_present: torch.Tensor,     # Complex wave tensor [Ny, Nx]
        v_causal: torch.Tensor,        # Causal velocity field [2, Ny, Nx]
        event_mask_x: torch.Tensor,    # Target event X mask [Ny, Nx]
        dt: float = 0.02,
        rewind_steps: int = 20,
        forward_steps: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Rewinds psi_present back to t_0 (+v), applies e^(iπ) phase inversion to event X,
        and re-forwards to t_1 (-v). Returns (psi_counterfactual, psi_rewound, causal_impact).
        """
        dev = psi_present.device
        psi_curr = psi_present.clone()

        # Step 1: Rewind time-reversed advection (+v)
        for _ in range(rewind_steps):
            dpsi_dx = (torch.roll(psi_curr, -1, dims=1) - torch.roll(psi_curr, 1, dims=1)) / 2.0
            dpsi_dy = (torch.roll(psi_curr, -1, dims=0) - torch.roll(psi_curr, 1, dims=0)) / 2.0
            time_reversed_advection = +(v_causal[0] * dpsi_dx + v_causal[1] * dpsi_dy)
            psi_curr = psi_curr + time_reversed_advection * dt

        psi_rewound = psi_curr.clone()

        # Step 2: Phase inversion e^(iπ) = -1.0 on event X
        psi_event_x = psi_curr * event_mask_x
        phase_inversion_factor = torch.complex(torch.tensor(-1.0, device=dev), torch.tensor(0.0, device=dev))
        psi_modified = psi_curr + (phase_inversion_factor * psi_event_x)

        # Step 3: Re-forward advection (-v) to counterfactual future
        psi_counterfactual = psi_modified.clone()
        for _ in range(forward_steps):
            dpsi_dx = (torch.roll(psi_counterfactual, -1, dims=1) - torch.roll(psi_counterfactual, 1, dims=1)) / 2.0
            dpsi_dy = (torch.roll(psi_counterfactual, -1, dims=0) - torch.roll(psi_counterfactual, 1, dims=0)) / 2.0
            forward_advection = -(v_causal[0] * dpsi_dx + v_causal[1] * dpsi_dy)

            density = torch.abs(psi_counterfactual) ** 2
            attractor = 0.05 * psi_counterfactual * (1.0 - density)
            psi_counterfactual = psi_counterfactual + (forward_advection + attractor) * dt

        causal_impact = float(torch.norm(psi_present - psi_counterfactual).item())
        return psi_counterfactual, psi_rewound, causal_impact


class KuramotoDualFrameCoupler(nn.Module):
    """
    Kuramoto Dual-Frame Coupler & Bivector Deadlock Unlocker.
    Couples Self (G_self) and Other/Human (G_other) frames without cross-contamination.
    Detects 180° (π rad) topological deadlocks and injects orthogonal bivector rotors
    to break saddle points and guarantee Lyapunov monotonic energy decay V(t) -> 0.
    """

    def __init__(
        self,
        dim: int = 2,
        eta: float = 0.25,
        gamma: float = 0.08,
        beta: float = 0.40,
        device: Optional[str] = None
    ):
        super().__init__()
        self.dim = dim
        self.eta = eta
        self.gamma = gamma
        self.beta = beta

        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dev = torch.device(self.device_str)

        self.register_buffer("G_base", torch.eye(dim, dtype=torch.float32, device=dev))
        self.register_buffer("G", torch.eye(dim, dtype=torch.float32, device=dev))

    def compute_lyapunov_energy(self, theta_obs: float, theta_target: float) -> float:
        """
        Calculates Lyapunov candidate energy function V(t):
        V(t) = 1/2 * (Δθ)^2 + (β / (2 * η)) * ||G - G_0||_F^2
        """
        delta_theta = theta_obs - theta_target
        G_diff = self.G - self.G_base
        norm_sq = torch.norm(G_diff, p="fro").item() ** 2
        V_t = 0.5 * (delta_theta ** 2) + (self.beta / (2.0 * self.eta)) * norm_sq
        return float(V_t)

    def unlock_topological_deadlock(
        self,
        psi1: torch.Tensor,
        psi2: torch.Tensor,
        epsilon_deadlock: float = 0.08,
        bivector_theta: float = 0.15,
        coupling_k: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Detects 180° (π rad) topological deadlock (|Δφ| ≈ π, torque ≈ 0)
        and injects orthogonal bivector rotation perturbation to resume gradient flow.
        Returns: (psi2_unlocked, effective_torque, is_deadlock_detected)
        """
        dev = psi1.device
        phi1 = torch.angle(psi1)
        phi2 = torch.angle(psi2)

        delta_phi = phi2 - phi1
        # Normalize to [-pi, pi]
        delta_phi = torch.remainder(delta_phi + math.pi, 2 * math.pi) - math.pi

        dist_from_pi = torch.abs(torch.abs(delta_phi) - math.pi)
        is_deadlock = bool(torch.mean(dist_from_pi).item() < epsilon_deadlock)

        psi2_unlocked = psi2.clone()
        if is_deadlock:
            # Inject bivector rotation perturbation
            asymmetry = torch.where(
                torch.arange(psi2.numel(), device=dev).reshape(psi2.shape) % 2 == 0,
                1.0, -1.0
            )
            phi2_new = phi2 + asymmetry * bivector_theta
            mag2 = torch.abs(psi2)
            psi2_unlocked = torch.complex(mag2 * torch.cos(phi2_new), mag2 * torch.sin(phi2_new))

        new_delta = torch.angle(psi2_unlocked) - phi1
        effective_torque = coupling_k * torch.sin(new_delta)
        return psi2_unlocked, effective_torque, is_deadlock

    def step_coupling(
        self,
        theta_obs: float,
        theta_target: float,
        dt: float = 0.05
    ) -> Tuple[float, float, float]:
        """
        Advances 1 step of phase locking and metric evolution.
        Returns: (new_theta_obs, G_01, lyapunov_V)
        """
        dev = self.G.device
        delta_theta = theta_obs - theta_target

        shear_tensor = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=dev, dtype=torch.float32)
        dG_dt = self.eta * delta_theta * shear_tensor - self.gamma * (self.G - self.G_base)

        with torch.no_grad():
            self.G += dG_dt * dt
            self.G = 0.5 * (self.G + self.G.t())

        dtheta_dt = -self.beta * self.G[0, 1].item()
        new_theta_obs = theta_obs + dtheta_dt * dt

        lyapunov_V = self.compute_lyapunov_energy(new_theta_obs, theta_target)
        return new_theta_obs, self.G[0, 1].item(), lyapunov_V


class ScaleRenormalizationEngine(nn.Module):
    """
    4D Scale-Space (x, y, z, s) Renormalization Group (RG) Engine.
    Executes Wilsonian scale coarse-graining (bottom-up emergence),
    Clifford bivector grade promotion (v_i ∧ v_j -> B), and
    HJB back-projection top-down value constraint torque.
    """

    def __init__(
        self,
        num_scales: int = 4,
        spatial_dim: int = 16,
        beta_topdown: float = 0.3,
        device: Optional[str] = None
    ):
        super().__init__()
        self.num_scales = num_scales
        self.spatial_dim = spatial_dim
        self.beta_topdown = beta_topdown

        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dev = torch.device(self.device_str)

        # Scale-space field [NumScales, Spatial, Spatial, 8]
        self.register_buffer(
            "scale_field",
            torch.zeros((num_scales, spatial_dim, spatial_dim, 8), dtype=torch.float32, device=dev)
        )

    def coarse_grain_step(self, s: int) -> torch.Tensor:
        """
        Bottom-up Wilsonian coarse-graining step from scale s to s+1.
        Promotes vector collisions (v_i ∧ v_j) into bivector rotation planes.
        """
        if s >= self.num_scales - 1:
            return self.scale_field[s]

        curr_layer = self.scale_field[s]
        # Smooth spatial field (scale kernel)
        v1 = curr_layer[..., 1]
        v2 = curr_layer[..., 2]

        # Bivector promotion: B_12 = v1 ∧ v2
        bivector_b12 = v1 * torch.roll(v2, shifts=1, dims=0) - v2 * torch.roll(v1, shifts=1, dims=0)

        next_layer = curr_layer.clone()
        next_layer[..., 0] = (curr_layer[..., 0] + torch.roll(curr_layer[..., 0], shifts=1, dims=0)) * 0.5
        next_layer[..., 4] += bivector_b12 * 0.2

        with torch.no_grad():
            self.scale_field[s + 1] = next_layer

        return self.scale_field[s + 1]

    def apply_topdown_constraint(self, dt: float = 0.05):
        """
        HJB back-projection: Top-down value constraint from s_max to lower scales s.
        """
        macro_value_grad = self.scale_field[-1, ..., 0] - self.scale_field[-1, ..., 0].mean()
        for s in range(self.num_scales - 1):
            penetration = math.exp(-0.5 * (self.num_scales - 1 - s))
            torque = self.beta_topdown * penetration * macro_value_grad
            with torch.no_grad():
                self.scale_field[s, ..., 1] += torque * dt
                self.scale_field[s, ..., 2] -= torque * dt
