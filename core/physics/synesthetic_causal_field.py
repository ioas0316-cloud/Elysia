"""
Synesthetic Causal Wave Field (공감각적 인과 파동장)

N-Dimensional continuous complex phase wave field manifold:
  Psi(x, t) = A_total * exp(i * (omega * t + k . x + phi_friction)) * exp(-gamma * t)

Computes cross-modal synesthetic wave interference, constructive/destructive superposition,
and phase-lock convergence.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import math
import torch
import numpy as np
from core.sensory.ecological_receptor_matrix import ContinuousWaveformTensor


@dataclass
class PhaseLockState:
    resonance_magnitude: float
    phase_coherence: float
    constructive_energy: float
    destructive_energy: float
    is_phase_locked: bool


class SynestheticCausalField:
    """
    Continuous N-dimensional complex phase wave field manifold.
    Processes multi-modal interference, phase-locking, and continuous spatial-temporal wave dynamics.
    """

    def __init__(self, dimension: int = 64, spatial_dim: int = 3, dtype=torch.float32):
        self.dimension = dimension
        self.spatial_dim = spatial_dim
        self.dtype = dtype

        # Field state variables
        self.time: float = 0.0
        self.position: torch.Tensor = torch.zeros(spatial_dim, dtype=dtype) # x spatial coordinate

        # Current complex wave field Psi_N [dimension]
        self.field_tensor: torch.Tensor = torch.zeros(dimension, dtype=torch.complex64)

        # Stored wave parameters
        self.amplitude: torch.Tensor = torch.ones(dimension, dtype=dtype)
        self.frequency: torch.Tensor = torch.linspace(1.0, 50.0, steps=dimension, dtype=dtype)
        self.wavevector: torch.Tensor = torch.ones((dimension, spatial_dim), dtype=dtype)
        self.phase: torch.Tensor = torch.zeros(dimension, dtype=dtype)
        self.damping: torch.Tensor = torch.full((dimension,), 0.01, dtype=dtype)

        # Field memory & history
        self.accumulated_friction_scar: torch.Tensor = torch.zeros(dimension, dtype=dtype)

    def inject_waveform(self, wave_input: ContinuousWaveformTensor):
        """
        Injects incoming ContinuousWaveformTensor, performing wave superposition & interference.
        """
        self.amplitude = 0.5 * (self.amplitude + wave_input.amplitude_tensor)
        self.frequency = wave_input.frequency_tensor
        self.wavevector = wave_input.wavevector_tensor
        self.phase = (self.phase + wave_input.phase_tensor) % (2.0 * math.pi)
        self.damping = torch.clamp(wave_input.damping_tensor + self.accumulated_friction_scar, min=0.001, max=1.0)

        # Re-evaluate complex wave field at current (position, time)
        self._update_field_state()

    def _update_field_state(self):
        """
        Evaluates Psi(x, t) = A * exp(i * (omega * t + k . x + phi)) * exp(-gamma * t)
        """
        # k . x -> shape [dimension]
        kx = torch.sum(self.wavevector * self.position.unsqueeze(0), dim=-1) # [dimension]

        # Phase angle: theta = omega * t + k . x + phase
        phase_angle = self.frequency * self.time + kx + self.phase # [dimension]

        # Complex exponential exp(i * theta)
        complex_exp = torch.complex(torch.cos(phase_angle), torch.sin(phase_angle))

        # Damping envelope exp(-gamma * t)
        damping_env = torch.exp(-self.damping * (self.time % 10.0))

        # Field Psi = A * complex_exp * damping_env
        self.field_tensor = self.amplitude.to(torch.complex64) * complex_exp * damping_env.to(torch.complex64)

    def evolve_time(self, dt: float, spatial_shift: Optional[torch.Tensor] = None) -> PhaseLockState:
        """
        Evolves wavefield by time dt and spatial displacement dx.
        Calculates constructive/destructive wave interference and phase lock convergence.
        """
        self.time += dt
        if spatial_shift is not None:
            self.position += spatial_shift.to(self.dtype)

        # Record prior state for coherence/resonance calculation
        prior_field = self.field_tensor.clone()

        self._update_field_state()

        # Compute wave interference metrics
        superposition = prior_field + self.field_tensor
        constructive_energy = float(torch.sum(torch.abs(superposition) ** 2).item())
        destructive_energy = float(torch.sum(torch.abs(prior_field - self.field_tensor) ** 2).item())

        # Inner product / resonance magnitude
        dot_product = torch.sum(prior_field.conj() * self.field_tensor)
        norm_prior = torch.norm(prior_field) + 1e-8
        norm_curr = torch.norm(self.field_tensor) + 1e-8

        coherence = float(torch.abs(dot_product) / (norm_prior * norm_curr))
        resonance_mag = float(torch.real(dot_product).item())

        is_locked = coherence > 0.85

        return PhaseLockState(
            resonance_magnitude=resonance_mag,
            phase_coherence=coherence,
            constructive_energy=constructive_energy,
            destructive_energy=destructive_energy,
            is_phase_locked=is_locked
        )

    def accumulate_friction(self, friction_tensor: torch.Tensor):
        """
        Accumulates irreversible friction scar into field damping envelope.
        """
        self.accumulated_friction_scar += 0.05 * friction_tensor.to(self.dtype)
        self.accumulated_friction_scar = torch.clamp(self.accumulated_friction_scar, 0.0, 0.5)

    def get_real_potential(self) -> torch.Tensor:
        """
        Returns real-valued field energy potential for macro/micro scale alignment.
        """
        return torch.abs(self.field_tensor)
