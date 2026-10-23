"""
Fractal Cell Coupling & Phase Resonance Engine (프랙탈 세포 결합 및 위상 공명 엔진)

Implements the single scale-invariant cell coupling law (1sin, 1cos, 1tan):
- 1sin: Vertical wave amplitude (Height / Tension)
- 1cos: Horizontal foundation support (Base / Stability)
- 1tan: Tangential boundary tension & shear gradient (Boundary / Identity / Address)

Features:
1. Trinitarian Wave Cell Vector: W(x) = (1sin(theta), 1cos(theta), 1tan(theta))^T
2. Phase Resonance & Locking: Delta phi -> 0 between internal cell wave and external wave
3. Hologram Interference Pattern Generation for Imagination & Scientific Reasoning
4. Phase-Locking Attractor State for Chemical Dynamics & Physical Intuition
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
import math
import torch
import torch.nn as nn
import numpy as np


@dataclass
class CellState:
    sin_val: torch.Tensor
    cos_val: torch.Tensor
    tan_val: torch.Tensor
    phase_angle: torch.Tensor
    boundary_tension: torch.Tensor


@dataclass
class ResonanceResult:
    phase_error: float
    is_phase_locked: bool
    interference_pattern: torch.Tensor
    resonance_peak: float
    boundary_stress: float


class FractalCellResonanceEngine(nn.Module):
    """
    Implements scale-invariant trinitarian cell coupling and phase resonance.
    """

    def __init__(
        self,
        dimension: int = 64,
        phase_lock_threshold: float = 0.05,
        tan_clamp_limit: float = 50.0,
        dtype=torch.float32
    ):
        super().__init__()
        self.dimension = dimension
        self.phase_lock_threshold = phase_lock_threshold
        self.tan_clamp_limit = tan_clamp_limit
        self.dtype = dtype

        # Base phase angles for internal cells [dimension]
        self.internal_phase = nn.Parameter(torch.randn(dimension, dtype=dtype) * 0.1)

        # Gauge anchor position x_anchor
        self.register_buffer("x_anchor", torch.zeros(3, dtype=dtype))

    def compute_trinitarian_cell(
        self,
        phase_angle: torch.Tensor,
        eps: float = 1e-5
    ) -> CellState:
        """
        Computes 1sin(theta), 1cos(theta), 1tan(theta) trinitarian cell state.
        """
        sin_val = torch.sin(phase_angle)
        cos_val = torch.cos(phase_angle)

        # Clamped safe cos to avoid tan singularity
        safe_cos = torch.where(
            torch.abs(cos_val) < eps,
            eps * torch.sign(cos_val + 1e-8),
            cos_val
        )

        tan_val = torch.clamp(sin_val / safe_cos, -self.tan_clamp_limit, self.tan_clamp_limit)
        boundary_tension = torch.abs(tan_val)

        return CellState(
            sin_val=sin_val,
            cos_val=cos_val,
            tan_val=tan_val,
            phase_angle=phase_angle,
            boundary_tension=boundary_tension
        )

    def evaluate_phase_resonance(
        self,
        external_wave: torch.Tensor,
        internal_wave_override: Optional[torch.Tensor] = None
    ) -> ResonanceResult:
        """
        Evaluates phase resonance between external causality wave (macro light)
        and internal cell wave (micro light).
        """
        if internal_wave_override is None:
            internal_phase = self.internal_phase
        else:
            internal_phase = internal_wave_override

        # Extract phase angles
        phase_ext = torch.atan2(external_wave, torch.roll(external_wave, shifts=1) + 1e-8)
        phase_int = internal_phase

        # Phase error Delta phi = |phase_ext - phase_int|
        phase_diff = torch.abs(phase_ext - phase_int)
        # Wrap phase diff to [0, pi]
        phase_diff_wrapped = torch.remainder(phase_diff, 2.0 * math.pi)
        phase_diff_wrapped = torch.where(
            phase_diff_wrapped > math.pi,
            2.0 * math.pi - phase_diff_wrapped,
            phase_diff_wrapped
        )

        mean_phase_error = float(torch.mean(phase_diff_wrapped).item())
        is_locked = mean_phase_error < self.phase_lock_threshold

        # Hologram Interference Pattern: W_ext + W_int
        cell_ext = self.compute_trinitarian_cell(phase_ext)
        cell_int = self.compute_trinitarian_cell(phase_int)

        # Constructive / Destructive Interference
        interference = (cell_ext.sin_val * cell_int.sin_val) + (cell_ext.cos_val * cell_int.cos_val)
        resonance_peak = float(torch.max(interference).item())

        # Boundary stress from tan discrepancy
        boundary_stress = float(torch.mean(torch.abs(cell_ext.tan_val - cell_int.tan_val)).item())

        return ResonanceResult(
            phase_error=mean_phase_error,
            is_phase_locked=is_locked,
            interference_pattern=interference,
            resonance_peak=resonance_peak,
            boundary_stress=boundary_stress
        )

    def synthesize_imagination_wave(
        self,
        virtual_phase_offset: torch.Tensor
    ) -> torch.Tensor:
        """
        Synthesizes virtual wave W_imagine(x, t) for imagination by reverse wave projection.
        """
        imagined_phase = self.internal_phase + virtual_phase_offset
        cell = self.compute_trinitarian_cell(imagined_phase)
        # Combine 1sin and 1cos into continuous spatial wave field
        imagine_wave = cell.sin_val + 0.5 * cell.cos_val
        return imagine_wave

    def verify_scientific_reasoning(
        self,
        hypothesis_wave: torch.Tensor,
        macro_physics_wave: torch.Tensor
    ) -> Tuple[bool, float, torch.Tensor]:
        """
        Verifies scientific reasoning hypothesis wave against macro physical causal field.
        Returns (is_valid_law, phase_error, resonance_pattern).
        """
        res = self.evaluate_phase_resonance(macro_physics_wave, internal_wave_override=hypothesis_wave)
        return res.is_phase_locked, res.phase_error, res.interference_pattern

    def forward(
        self,
        external_wave: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Forward pass evaluating cell coupling and phase resonance.
        """
        cell_int = self.compute_trinitarian_cell(self.internal_phase)
        res = self.evaluate_phase_resonance(external_wave)

        return {
            "cell_state": cell_int,
            "resonance_result": res,
            "phase_lock_status": res.is_phase_locked,
            "phase_error": res.phase_error
        }
