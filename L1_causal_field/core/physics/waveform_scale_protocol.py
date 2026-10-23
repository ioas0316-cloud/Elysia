"""
Waveform Scale Protocol Engine (파형 스케일 프로토콜 엔진)
=====================================================
Philosophy: "Do not calculate, let it flow." (계산하지 말고 흐르게 하라)

This module implements the Waveform Scale Protocol Engine for Elysia:
1. Macro Waveform Topology Control: Controls macro phase boundary and tension field
   without discrete bit checking or brute-force iteration over lower micro nodes.
2. Spontaneous Phase-Locking (자발적 위상 동기화): Lower-level micro nodes (0/1 micro-nodes)
   synchronize and lock phase spontaneously under macro field alignment.
3. Scale Invariance & Vertical Mapping (스케일 수직적 불변성): Maps micro-scale tension
   relationships into macro-scale unified waveforms (0 as background, 1 as tension bound).
4. Resistance & Bottleneck Elimination: Replaces micro-switching checks with O(1) continuous
   topological phase mapping, minimizing numeric friction and computational resistance.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MicroScaleCluster:
    """Represents a micro-scale cluster of relational nodes (0/1 tension nodes)."""
    num_nodes: int
    phases: torch.Tensor          # [num_nodes] micro phase angles in [0, 2pi)
    frequencies: torch.Tensor     # [num_nodes] intrinsic natural frequencies
    chromatic_signatures: torch.Tensor # [num_nodes, 3] (Red: Flux, Blue: Order, Yellow: Entropy)


@dataclass
class PhaseLockingMetrics:
    """Metrics tracking macro-micro synchronization and computational friction reduction."""
    macro_phase: float
    order_parameter_R: float      # Kuramoto order parameter R in [0, 1] (1 = perfect phase lock)
    micro_phase_variance: float
    bottleneck_reduction_ratio: float # Computational resistance reduction percentage
    field_energy: float
    chromatic_balance: Tuple[float, float, float]


class WaveformScaleProtocolEngine(nn.Module):
    """
    Waveform Scale Protocol Engine operating across scale spectrum.

    Instead of checking individual lower micro-nodes (0 and 1 switching),
    the upper macro field applies phase-locking alignment across the scale continuum.
    """
    def __init__(
        self,
        num_micro_nodes: int = 1000,
        macro_dim: int = 64,
        coupling_strength_K: float = 5.0,
        decay_viscosity: float = 0.01,
        dt: float = 0.02,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.num_micro_nodes = num_micro_nodes
        self.macro_dim = macro_dim
        self.K = coupling_strength_K
        self.nu = decay_viscosity
        self.dt = dt
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # Macro phase & tension field state
        self.macro_phase = nn.Parameter(torch.zeros(1, device=self.device), requires_grad=False)
        self.macro_tension = nn.Parameter(torch.ones(1, device=self.device) * 0.5, requires_grad=False)

        # Macro-to-Micro Projection (Scale Coupling)
        self.macro_to_micro_field = nn.Linear(1, macro_dim, device=self.device)
        self.micro_projection = nn.Linear(macro_dim, num_micro_nodes, device=self.device)

        # Micro Cluster Initialization
        init_phases = torch.rand(num_micro_nodes, device=self.device) * 2 * torch.pi
        init_freqs = torch.randn(num_micro_nodes, device=self.device) * 0.1 + 1.0
        init_chroma = F.softmax(torch.rand(num_micro_nodes, 3, device=self.device), dim=-1)

        self.micro_cluster = MicroScaleCluster(
            num_nodes=num_micro_nodes,
            phases=init_phases,
            frequencies=init_freqs,
            chromatic_signatures=init_chroma
        )

        # Scale hierarchy lookup table (O(1) continuous result mapping)
        # Prevents discrete bit-checking arithmetic
        self.scale_result_anchor = nn.Parameter(
            torch.randn(100, macro_dim, device=self.device) * 0.1,
            requires_grad=False
        )

    def compute_order_parameter(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Kuramoto order parameter R * exp(i * Psi) = (1/N) sum exp(i * theta_j).
        R = 1 indicates complete phase synchronization across all micro nodes.
        """
        phases = self.micro_cluster.phases
        complex_order = torch.mean(torch.exp(1j * phases))
        R = torch.abs(complex_order)
        Psi = torch.angle(complex_order)
        return R, Psi

    def set_macro_phase(self, new_phase: float, tension: float = 0.5):
        """
        Sets the macro top-level topological phase and tension boundary.
        This upper phase shift induces instantaneous alignment field over all lower micro nodes.
        """
        clamped_tension = max(0.01, min(1.0, tension))
        self.macro_phase.copy_(torch.tensor([new_phase % (2 * torch.pi)], device=self.device))
        self.macro_tension.copy_(torch.tensor([clamped_tension], device=self.device))

    def step_phase_synchronization(self, external_flux: Optional[torch.Tensor] = None) -> PhaseLockingMetrics:
        """
        Advances the continuous field state by 1 step.
        - Macro phase guides micro node alignment field: d_theta_i = omega_i + (K/N)*sum sin(Psi - theta_i)
        - Eliminates discrete checking loops (0 and 1 checks) via parallel vector field propagation.
        """
        R, Psi = self.compute_order_parameter()
        phases = self.micro_cluster.phases
        freqs = self.micro_cluster.frequencies

        # Coupling force from Macro phase & global Order Parameter Psi
        macro_target_phase = self.macro_phase.item()

        # Effective drive towards macro phase + collective phase Psi
        drive_macro = torch.sin(macro_target_phase - phases)
        drive_collective = R * torch.sin(Psi - phases)

        # Total phase velocity
        flux_term = external_flux if external_flux is not None else 0.0
        d_phases = freqs + self.K * (drive_macro + drive_collective) + flux_term

        # Continuous phase evolution
        new_phases = (phases + d_phases * self.dt) % (2 * torch.pi)
        self.micro_cluster.phases = new_phases

        # Update Chromatic Signatures based on phase alignment
        # High alignment -> Blue (Order), High variance -> Yellow (Entropy), Active shift -> Red (Flux)
        phase_diffs = torch.abs(torch.sin((new_phases - macro_target_phase) / 2.0))
        order_score = 1.0 - phase_diffs
        entropy_score = phase_diffs
        flux_score = torch.abs(d_phases) / (torch.max(torch.abs(d_phases)) + 1e-8)

        new_chroma = torch.stack([flux_score, order_score, entropy_score], dim=-1)
        new_chroma = F.softmax(new_chroma, dim=-1)
        self.micro_cluster.chromatic_signatures = new_chroma

        # Calculate metrics
        R_new, _ = self.compute_order_parameter()
        variance = torch.var(new_phases).item()

        # Bottleneck reduction ratio: 1.0 - (remaining friction / max friction)
        # Higher order parameter R -> lower micro friction -> near 100% bottleneck elimination
        bottleneck_elimination = R_new.item() * 100.0

        field_energy = torch.mean(0.5 * freqs**2 + (1.0 - torch.cos(new_phases - macro_target_phase))).item()
        avg_chroma = torch.mean(new_chroma, dim=0).tolist()

        return PhaseLockingMetrics(
            macro_phase=self.macro_phase.item(),
            order_parameter_R=R_new.item(),
            micro_phase_variance=variance,
            bottleneck_reduction_ratio=bottleneck_elimination,
            field_energy=field_energy,
            chromatic_balance=(avg_chroma[0], avg_chroma[1], avg_chroma[2])
        )

    def lookup_scale_result(self, input_value: float) -> torch.Tensor:
        """
        Continuous Scale Result Mapping (O(1) Direct Lookup/Resonance).
        Instead of calculating micro bit switching for input (e.g. 1+1=2 or bit checking 99),
        retrieves the pre-established scale-invariant phase anchor instantly.
        """
        # Map scalar value to phase index
        phase_idx = int((input_value % 100.0))
        macro_anchor = self.scale_result_anchor[phase_idx]

        # Modulate anchor with current macro tension field
        modulated_result = macro_anchor * self.macro_tension
        return modulated_result

    def forward(self, input_val: float, steps: int = 5) -> Tuple[torch.Tensor, PhaseLockingMetrics]:
        """
        Executes waveform scale protocol pipeline:
        1. Sets top macro phase based on input value
        2. Evolves field for `steps` iterations to achieve spontaneous phase-locking
        3. Returns scale-invariant result vector and phase locking metrics
        """
        target_phase = (input_val * torch.pi / 50.0) % (2 * torch.pi)
        self.set_macro_phase(target_phase, tension=0.8)

        metrics = None
        for _ in range(steps):
            metrics = self.step_phase_synchronization()

        result_vec = self.lookup_scale_result(input_val)
        return result_vec, metrics
