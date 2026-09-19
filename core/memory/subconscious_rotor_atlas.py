"""
Subconscious SSD Atlas & Static Rotor Aux-Memory Unit (SRAM-R).

This module implements:
1. StaticRotorAuxMemoryUnit (SRAM-R):
   - Rotor Tag Registers (RTR): Holds relative phase offset bivectors Delta R / Omega.
   - Phase Delta Calculator (PDC): Calculates Delta R = R_curr * R_base~ during eviction.
   - Rotor Alignment Engine (RAE): Re-applies preserved Delta R during fetch to eliminate cold start/phase jump.
   - Freeze/Sync phase management.

2. SubconsciousSSDAtlas & LocalChart:
   - LocalChart: Represents local spatiotemporal memory charts with metric g_mem(x),
     hardening coefficient gamma, and decay/diffusion parameters.
   - SubconsciousSSDAtlas: Manages global chart atlas, event-driven consolidation (Phase-Lock triggering),
     natural entropy decay & diffusion, and phase-locked hardening (gamma -> epsilon).
"""

import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn


class StaticRotorAuxMemoryUnit(nn.Module):
    """
    Static Rotor Aux-Memory Unit (SRAM-R).
    Holds phase frame invariants across memory hierarchy swaps (Cache <-> DRAM/SSD).
    """

    def __init__(self, state_dim: int, max_tags: int = 1024):
        super().__init__()
        self.state_dim = state_dim
        self.max_tags = max_tags

        # Rotor Tag Register (RTR) storing skew-symmetric bivector matrices Omega in Lie Algebra so(N)
        self.rtr_store: Dict[str, torch.Tensor] = {}
        # RTR background drift accumulation rate
        self.bg_drift_rate = 0.001

    def freeze_phase(self, tag_id: str, Q_curr: torch.Tensor, Q_base: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        [PDC - Phase Delta Calculator] (Eviction / Freeze Phase):
        Calculates relative phase offset Delta Q = Q_curr * Q_base^T and extracts Lie algebra
        bivector Omega = - log(Delta Q) / dt.
        """
        B, N, _ = Q_curr.shape
        if Q_base is None:
            Q_base = torch.eye(N, device=Q_curr.device, dtype=Q_curr.dtype).unsqueeze(0).repeat(B, 1, 1)

        # Delta Q = Q_curr * Q_base^T
        Delta_Q = torch.bmm(Q_curr, Q_base.transpose(-1, -2))

        # Extract skew-symmetric Lie algebra bivector via matrix log approximation or skew projection
        # Omega = 0.5 * (Delta_Q - Delta_Q^T)
        Omega_skew = 0.5 * (Delta_Q - Delta_Q.transpose(-1, -2))

        # Store in RTR
        self.rtr_store[tag_id] = Omega_skew.detach().clone()
        return Omega_skew

    def update_background_drift(self, dt: float = 0.01):
        """
        Background phase drift accumulation during idle/subconscious hold state.
        Delta R_{t+dt} = Delta R_t * exp(-dt * Omega_bg)
        """
        for tag_id, Omega in self.rtr_store.items():
            drift_tensor = torch.randn_like(Omega) * self.bg_drift_rate
            drift_skew = 0.5 * (drift_tensor - drift_tensor.transpose(-1, -2))
            self.rtr_store[tag_id] = Omega + dt * drift_skew

    def sync_phase(self, tag_id: str, Q_global: torch.Tensor, dt_step: float = 0.04) -> torch.Tensor:
        """
        [RAE - Rotor Alignment Engine] (Fetch / Sync Phase):
        Re-applies preserved phase delta Q_restored = Q_global * exp(-dt * Omega_rtr)
        restoring exact phase continuity upon cache line fetch.
        """
        if tag_id not in self.rtr_store:
            return Q_global

        Omega_rtr = self.rtr_store[tag_id].to(Q_global.device, dtype=Q_global.dtype)
        rot_step = torch.matrix_exp(-dt_step * Omega_rtr)
        Q_restored = torch.bmm(Q_global, rot_step)
        return Q_restored


class LocalChart:
    """
    Spatiotemporal Local Chart in Subconscious Memory Atlas.
    """

    def __init__(
        self,
        chart_id: str,
        center: torch.Tensor,
        state_dim: int,
        decay_rate: float = 0.05,
        diffusion_coeff: float = 0.01,
    ):
        self.chart_id = chart_id
        self.center = center.detach().clone()  # [state_dim]
        self.state_dim = state_dim

        # Metric tensor g_mem (initialized to Euclidean Identity I)
        self.g_mem = torch.eye(state_dim, device=center.device, dtype=center.dtype)

        # Decay/hardening coefficient gamma (gamma -> epsilon when hardened)
        self.gamma = decay_rate
        self.diffusion_coeff = diffusion_coeff
        self.lock_count = 0
        self.is_hardened = False

    def apply_consolidation(self, v_shift: torch.Tensor, Omega_skew: torch.Tensor, plasticity_rate: float = 0.05):
        """
        Consolidates dynamic cache sampling trajectory into local metric g_mem:
        Delta g_mem = plasticity_rate * (v_shift (x) v_shift^T + Omega_skew)
        """
        v_outer = torch.mm(v_shift.unsqueeze(1), v_shift.unsqueeze(0))  # [N, N]
        distortion = v_outer + 0.5 * (Omega_skew + Omega_skew.T)

        self.g_mem = self.g_mem + plasticity_rate * distortion
        self.lock_count += 1

        # Phase-Locked Consolidation (Hardening when lock_count exceeds threshold)
        if self.lock_count >= 5:
            self.gamma = 1e-4  # gamma -> epsilon
            self.is_hardened = True

    def natural_decay_and_diffusion(self, dt: float = 0.1):
        """
        Natural entropy decay and curvature diffusion:
        dg_mem/dt = - gamma * (g_mem - I) + D * Laplacian(g_mem)
        """
        if self.is_hardened:
            return

        I_base = torch.eye(self.state_dim, device=self.g_mem.device, dtype=self.g_mem.dtype)

        # Natural decay towards base Euclidean metric
        decay_term = -self.gamma * (self.g_mem - I_base)

        # Curvature diffusion / smoothing
        diag_mean = torch.trace(self.g_mem) / self.state_dim
        diffusion_term = -self.diffusion_coeff * (self.g_mem - diag_mean * I_base)

        self.g_mem = self.g_mem + dt * (decay_term + diffusion_term)


class SubconsciousSSDAtlas(nn.Module):
    """
    Subconscious Non-Volatile Memory Atlas (Global Chart Atlas).
    """

    def __init__(self, state_dim: int, chart_radius: float = 1.5):
        super().__init__()
        self.state_dim = state_dim
        self.chart_radius = chart_radius
        self.charts: Dict[str, LocalChart] = {}

    def get_or_create_chart(self, x_coord: torch.Tensor) -> LocalChart:
        """
        Pinpoints or creates a local chart matching x_coord in state space.
        """
        x_flat = x_coord.squeeze()
        best_id = None
        min_dist = float("inf")

        for chart_id, chart in self.charts.items():
            dist = torch.norm(x_flat - chart.center).item()
            if dist < min_dist:
                min_dist = dist
                best_id = chart_id

        if best_id is not None and min_dist <= self.chart_radius:
            return self.charts[best_id]

        # Create new chart
        new_id = f"chart_{len(self.charts)}"
        new_chart = LocalChart(new_id, x_flat, self.state_dim)
        self.charts[new_id] = new_chart
        return new_chart

    def consolidate_event(
        self,
        x_evt: torch.Tensor,
        v_shift: torch.Tensor,
        Omega_skew: torch.Tensor,
        plasticity_rate: float = 0.05,
    ) -> LocalChart:
        """
        Phase-Lock Triggered Memory Consolidation Event.
        Inscribes dynamic cache trajectory into subconscious SSD chart g_mem.
        """
        chart = self.get_or_create_chart(x_evt)
        chart.apply_consolidation(v_shift.squeeze(0), Omega_skew.squeeze(0), plasticity_rate)
        return chart

    def step_subconscious_decay(self, dt: float = 0.1):
        """
        Applies asynchronous background weathering (entropy decay and diffusion) across all charts.
        """
        for chart in self.charts.values():
            chart.natural_decay_and_diffusion(dt)
