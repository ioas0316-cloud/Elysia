"""
Execution Manifold and Cross-Dimensional Recalibration Module.

This module formalizes how low-level byte/bit execution traces (micro memory dynamics)
fuse with continuous multi-modal physical streams (vision, audio, mechanics) and project onto
macro semantic attractors. It implements:
  1. AssemblyTraceVectorizer: Vectorization of raw assembly register and memory logs.
  2. MemoryCognitiveEngine: Micro-trajectory steering via cross-dimensional drift E_cross.
  3. GroundedCognitiveEngine: Fusion with multi-modal physical streams & barrier potentials.
  4. retrocausal_reinterpretation_step: Backward-time adjoint pass (T -> 0) to reclassify noise.
  5. SelfReferentialMetaEngine: Meta-cognitive self-evolution loop updating internal geometry.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class AssemblyTraceVectorizer:
    """
    Parses raw assembly execution frames and maps them into the
    [d_m (64) | d_phi (16) | d_s (48)] contiguous execution manifold tensor layout.
    """
    def __init__(self, d_m: int = 64, d_phi: int = 16, d_s: int = 48, window_size: int = 8):
        self.d_m = d_m
        self.d_phi = d_phi
        self.d_s = d_s
        self.window_size = window_size
        self.prev_registers: Optional[np.ndarray] = None
        self.prev_sp: Optional[int] = None

    def _compute_bit_entropy(self, reg_val: int) -> float:
        """Calculates normalized byte entropy for a 64-bit register."""
        bytes_arr = np.array([(reg_val >> (i * 8)) & 0xFF for i in range(8)], dtype=np.uint8)
        _, counts = np.unique(bytes_arr, return_counts=True)
        probs = counts / 8.0
        return float(-np.sum(probs * np.log2(probs + 1e-9)) / 3.0)  # Normalized [0, 1]

    def process_frame(self, frame: Dict[str, int], macro_target: torch.Tensor) -> torch.Tensor:
        """
        Input Frame:
            {
              'RIP': 0x401122, 'RSP': 0x7fff510,
              'RAX': 0x01, 'RBX': 0xff, 'RCX': 0x10, 'RDX': 0x00,
              'mem_addr': 0x7fff508, 'mem_write': 1
            }
        """
        rax = frame.get('RAX', 0)
        rbx = frame.get('RBX', 0)
        rcx = frame.get('RCX', 0)
        rdx = frame.get('RDX', 0)
        regs = np.array([rax, rbx, rcx, rdx], dtype=np.uint64)
        rip = frame.get('RIP', 0)
        rsp = frame.get('RSP', 0)

        # -------------------------------------------------------------
        # 1. Micro Memory Channel (d_m = 64)
        # -------------------------------------------------------------
        if self.prev_registers is None:
            self.prev_registers = regs
            self.prev_sp = rsp

        # Bit Mutation Rate (Hamming distance across registers)
        bit_flips = sum(bin(int(curr ^ prev)).count('1') for curr, prev in zip(regs, self.prev_registers))
        bit_mutation_rate = float(bit_flips) / (64.0 * len(regs))

        # Stack Delta & Entropy
        stack_delta = float(abs(rsp - self.prev_sp)) / 1024.0
        reg_entropies = [self._compute_bit_entropy(int(r)) for r in regs]
        stack_entropy = float(np.mean(reg_entropies))

        # Access Stride Velocity
        mem_stride = float(abs(frame.get('mem_addr', rsp) - rsp)) / 4096.0
        write_flag = float(frame.get('mem_write', 0))

        # Register Flux vector (60 features)
        reg_diffs = (regs.astype(np.float64) - self.prev_registers.astype(np.float64)) / (2**32)
        reg_flux = np.pad(reg_diffs, (0, max(0, self.d_m - 4 - len(reg_diffs))), 'constant')[:self.d_m - 4]

        m_delta = np.concatenate([
            [bit_mutation_rate, stack_entropy, mem_stride, write_flag],
            reg_flux
        ]).astype(np.float32)

        # -------------------------------------------------------------
        # 2. Phase Tensor Channel (d_phi = 16)
        # -------------------------------------------------------------
        # Map Instruction Pointer (RIP) onto a cyclic phase angle [0, 2pi)
        loop_phase_angle = float((rip % 256) / 256.0 * (2.0 * np.pi))
        stack_depth_norm = float(min(1.0, float(rsp & 0xFFFF) / 65535.0))
        branch_entropy = float((rip >> 4) & 0x01)  # Proxy for branch state
        coherence = float(np.cos(loop_phase_angle))

        cadence = np.sin(np.linspace(0, loop_phase_angle, self.d_phi - 4)).astype(np.float32)

        phi_phase = np.concatenate([
            [loop_phase_angle, stack_depth_norm, branch_entropy, coherence],
            cadence
        ]).astype(np.float32)

        # -------------------------------------------------------------
        # Update State & Construct Full Vector
        # -------------------------------------------------------------
        self.prev_registers = regs
        self.prev_sp = rsp

        s_macro = macro_target.detach().cpu().numpy().flatten()
        if len(s_macro) != self.d_s:
            s_macro = np.pad(s_macro, (0, max(0, self.d_s - len(s_macro))), 'constant')[:self.d_s]

        z_exec = np.concatenate([m_delta, phi_phase, s_macro]).astype(np.float32)
        return torch.from_numpy(z_exec)


class MemoryCognitiveEngine(nn.Module):
    """
    Computes cross-dimensional drift E_cross between micro memory execution traces
    and macro symbol attractors, steering execution trajectory without spatial spatial metrics.
    """
    def __init__(self, d_m: int = 64, d_phi: int = 16, d_s: int = 48):
        super().__init__()
        self.d_m = d_m
        self.d_phi = d_phi
        self.d_s = d_s
        self.d_total = d_m + d_phi + d_s

        # Cross-Dimensional Projection Operator (Pi)
        self.proj_pi = nn.Sequential(
            nn.Linear(d_m + d_phi, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, d_s)
        )

    def forward(self, Z_exec: torch.Tensor, target_symbol_attractor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Z_exec: [Batch, D_total] - Execution State Vector Batch
        target_symbol_attractor: [Batch, D_s] - Intended Macro Symbol Attractor
        """
        m_delta = Z_exec[:, :self.d_m]
        phi_phase = Z_exec[:, self.d_m : self.d_m + self.d_phi]

        # 1. Project Micro Execution Traces to Macro Space
        micro_features = torch.cat([m_delta, phi_phase], dim=-1)
        s_inferred = self.proj_pi(micro_features)

        # 2. Compute Cross-Dimensional Drift (E_cross)
        semantic_drift = F.mse_loss(s_inferred, target_symbol_attractor, reduction='none').sum(dim=-1)
        noise_penalty = torch.norm(m_delta[:, :4], dim=-1)
        E_cross = semantic_drift + 0.05 * noise_penalty

        # 3. Micro Recalibration Gradient (Steering)
        grad_steering = torch.autograd.grad(E_cross.sum(), micro_features, create_graph=True)[0]
        micro_features_steered = micro_features - 0.01 * grad_steering

        Z_exec_next = torch.cat([micro_features_steered, s_inferred], dim=-1)
        return Z_exec_next, E_cross


class GroundedCognitiveEngine(nn.Module):
    """
    Integrates multi-modal physical streams (vision, audio, mechanics) directly into
    the micro-trace execution manifold, transforming execution states into physically-grounded
    cognitive trajectories.
    """
    def __init__(self, d_m: int = 64, d_v: int = 32, d_a: int = 16, d_mech: int = 16, d_phi: int = 16, d_s: int = 48):
        super().__init__()
        self.d_m = d_m
        self.d_v = d_v
        self.d_a = d_a
        self.d_mech = d_mech
        self.d_phi = d_phi
        self.d_s = d_s
        self.d_sensory = d_v + d_a + d_mech

        # Cross-Modal Fusion Encoder
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_m + self.d_sensory + d_phi, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, d_s)
        )

    def forward(
        self,
        z_m: torch.Tensor,
        x_v: torch.Tensor,
        x_a: torch.Tensor,
        x_mech: torch.Tensor,
        phi: torch.Tensor,
        target_s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate Micro Exec Traces with Physical Streams
        physical_state = torch.cat([z_m, x_v, x_a, x_mech, phi], dim=-1)
        if not physical_state.requires_grad:
            physical_state = physical_state.detach().requires_grad_(True)

        # Project Grounded State to Macro Symbol Attractor
        s_inferred = self.fusion_layer(physical_state)

        # Physical Barrier Penalty (Mechanics Force Limit + Vision Obstacle Proximity)
        force_limit_penalty = torch.relu(torch.norm(x_mech, dim=-1) - 1.0)
        obstacle_penalty = torch.relu(1.0 - torch.norm(x_v, dim=-1))

        # Cross-Dimensional Drift Energy
        e_drift = F.mse_loss(s_inferred, target_s, reduction='none').sum(dim=-1)
        e_grounded = e_drift + 0.2 * (force_limit_penalty + obstacle_penalty)

        # Compute Recalibration Steering Gradient
        grad_steering = torch.autograd.grad(e_grounded.sum(), physical_state, create_graph=True)[0]
        physical_state_steered = physical_state - 0.01 * grad_steering

        return physical_state_steered, e_grounded


def retrocausal_reinterpretation_step(
    z_trajectory: torch.Tensor,       # [T_steps, Batch, Dim]
    target_macro_s: torch.Tensor,     # [Batch, Dim_s]
    proj_operator: nn.Module,         # Pi operator or Grounded Engine fusion layer
    theta_causal: float = 0.15
) -> torch.Tensor:
    """
    Integrates backwards from T down to 0 to reclassify early sensory/execution noise
    into necessary precursor signals for the terminal macro goal.
    Returns precursor mask tensor of shape [T_steps - 1, Batch].
    """
    T_steps, batch_size, dim = z_trajectory.shape

    # 1. Terminal pass at step T
    z_T = z_trajectory[-1].detach().requires_grad_(True)
    s_inferred_T = proj_operator(z_T)
    c_terminal = 0.5 * torch.sum((s_inferred_T - target_macro_s)**2, dim=-1)

    # 2. Initialize Adjoint State Lambda at T
    lambda_t = torch.autograd.grad(c_terminal.sum(), z_T)[0]  # [Batch, Dim]

    reclassified_noise = []

    # 3. Backward Integration Cycle (T-1 down to 0)
    for t in reversed(range(T_steps - 1)):
        z_t = z_trajectory[t].detach().requires_grad_(True)
        z_next_pred = z_t + 0.01 * z_t  # Linear dynamical step approximation

        # Vector-Jacobian Product (VJP) backward step
        vjp = torch.autograd.grad(z_next_pred, z_t, grad_outputs=lambda_t, retain_graph=False)[0]
        lambda_t = vjp

        # 4. Score Early Noise against Retrocausal Sensitivity (Lambda)
        early_noise_t = z_trajectory[t] - z_trajectory[t - 1] if t > 0 else z_trajectory[0]
        kappa = torch.sum(early_noise_t * lambda_t, dim=-1)  # [Batch]

        is_precursor = (torch.abs(kappa) >= theta_causal).float()
        reclassified_noise.append(is_precursor)

    return torch.stack(reclassified_noise[::-1])


class SelfReferentialMetaEngine(nn.Module):
    """
    Self-referential meta-cognition engine that uses retrocausal sensitivity signals
    to rewrite its own internal manifold parameters theta_M, optimizing cognitive physics.
    """
    def __init__(self, d_m: int = 64, d_v: int = 32, d_a: int = 16, d_mech: int = 16, d_phi: int = 16, d_s: int = 48):
        super().__init__()
        self.grounded_engine = GroundedCognitiveEngine(d_m, d_v, d_a, d_mech, d_phi, d_s)
        self.meta_learning_rate = 1e-3

    def forward_trajectory(
        self,
        trajectory_inputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        target_s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs forward execution over a sequence of time steps.
        trajectory_inputs: list of (z_m, x_v, x_a, x_mech, phi) tuples.
        Returns z_trajectory [T_steps, Batch, Dim] and total drift energy.
        """
        steered_states = []
        total_drift = 0.0

        for z_m, x_v, x_a, x_mech, phi in trajectory_inputs:
            steered, drift = self.grounded_engine(z_m, x_v, x_a, x_mech, phi, target_s)
            steered_states.append(steered)
            total_drift = total_drift + drift.mean()

        z_trajectory = torch.stack(steered_states)
        return z_trajectory, total_drift

    def meta_update(
        self,
        z_trajectory: torch.Tensor,
        target_s: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Computes meta-cognitive parameter updates theta_M based on trajectory drift & retrocausal pass.
        """
        precursor_masks = retrocausal_reinterpretation_step(
            z_trajectory, target_s, self.grounded_engine.fusion_layer
        )

        # Meta loss penalized by non-precursor cognitive friction
        terminal_state = z_trajectory[-1]
        s_inferred = self.grounded_engine.fusion_layer(terminal_state)
        meta_loss = F.mse_loss(s_inferred, target_s) + 0.01 * (1.0 - precursor_masks.mean())

        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()

        return float(meta_loss.item())
