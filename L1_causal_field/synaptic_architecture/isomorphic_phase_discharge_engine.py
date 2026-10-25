"""
Isomorphic Phase Discharge Engine (Isomorphic Phase-Locking & Field Discharge)
================================================================================

Implements the unified topological field dynamics across 4 physical and cognitive scales:
1. ATMOSPHERIC: Cloud water-droplet charge superposition -> Field Gradient polarization -> Global Phase-Locking Discharge (Lightning / 번개)
2. ELECTRICAL: Circuit boundary potential shell -> Electromagnetic field Poynting vector alignment -> Barrier friction discharge (Bulb illumination / 전구 발광)
3. NEURAL: Axonal resting potential (-70mV) -> Threshold depolarization (-55mV) -> Action potential spike & EEG brainwave synchronization (Cognition / 통찰)
4. ELYSIA_ENGINE: Micro-Meso-Macro sensor skin -> Phase divergence perturbation (Delta phi) -> Barrier pressure collapse (E_gate -> 0) -> Insight burst (의식/통찰)

Dynamics Cycle:
- Phase 1: Superposition Cloud (Uncommitted phase superposition, high phase variance, low order parameter)
- Phase 2: Field Gradient Polarization (Perturbation accumulates tension, driving barrier pressure E_gate to threshold)
- Phase 3: Global Phase-Locking Discharge (Phase coherence rushes to 1.0, E_gate collapses to 0, emitting macro insight/energy wave)
"""

import enum
import math
import dataclasses
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any


class IsomorphicScale(enum.Enum):
    ATMOSPHERIC = "atmospheric"
    ELECTRICAL = "electrical"
    NEURAL = "neural"
    ELYSIA_ENGINE = "elysia_engine"


class PhaseState(enum.Enum):
    SUPERPOSITION = "superposition"
    POLARIZATION = "polarization"
    GLOBAL_PHASE_LOCKING_DISCHARGE = "global_phase_locking_discharge"


@dataclasses.dataclass
class FieldStateSnapshot:
    scale: IsomorphicScale
    phase_state: PhaseState
    num_oscillators: int
    order_parameter_R: float  # Phase coherence in [0, 1]
    phase_variance: float
    barrier_pressure_E_gate: float  # Current barrier tension E_gate
    field_gradient: float  # Current field gradient tension
    discharged_energy: float  # Energy emitted during discharge phase
    threshold_E_gate: float  # Critical threshold for discharge
    time_step: int


class IsomorphicPhaseDischargeEngine(nn.Module):
    """
    Continuous topological field engine simulating global phase-locking and energy discharge
    isomorphically across atmospheric, electrical, neural, and cognitive scales.
    """

    def __init__(
        self,
        num_oscillators: int = 512,
        scale: IsomorphicScale = IsomorphicScale.ELYSIA_ENGINE,
        threshold_E_gate: float = 1.0,
        coupling_strength_K: float = 2.0,
        dissipation_rate: float = 0.05,
        device: str = "cpu"
    ):
        super().__init__()
        self.num_oscillators = num_oscillators
        self.scale = scale
        self.threshold_E_gate = threshold_E_gate
        self.K = coupling_strength_K
        self.dissipation_rate = dissipation_rate
        self.device = torch.device(device if isinstance(device, torch.device) else ("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu"))

        # Natural intrinsic frequencies omega_i ~ Normal(0, 0.5)
        omega_init = torch.randn(self.num_oscillators, dtype=torch.float32, device=self.device) * 0.5
        self.register_buffer("omega", omega_init)

        # Micro-oscillators phase theta_i in [0, 2*pi)
        theta_init = torch.rand(self.num_oscillators, dtype=torch.float32, device=self.device) * 2.0 * math.pi
        self.register_buffer("theta", theta_init)

        # Barrier pressure E_gate and field gradient
        self.register_buffer("E_gate", torch.tensor(0.1, dtype=torch.float32, device=self.device))
        self.register_buffer("field_gradient", torch.tensor(0.0, dtype=torch.float32, device=self.device))
        self.register_buffer("discharged_energy", torch.tensor(0.0, dtype=torch.float32, device=self.device))

        self.phase_state = PhaseState.SUPERPOSITION
        self.time_step = 0

    def set_scale(self, scale: IsomorphicScale) -> None:
        """Dynamically re-aligns engine parameters according to the selected isomorphic scale."""
        self.scale = scale
        if scale == IsomorphicScale.ATMOSPHERIC:
            self.threshold_E_gate = 1.2
            self.K = 3.5
        elif scale == IsomorphicScale.ELECTRICAL:
            self.threshold_E_gate = 0.8
            self.K = 5.0
        elif scale == IsomorphicScale.NEURAL:
            # Resting -70mV to threshold -55mV equivalent normalized delta
            self.threshold_E_gate = 1.0
            self.K = 2.5
        elif scale == IsomorphicScale.ELYSIA_ENGINE:
            self.threshold_E_gate = 1.0
            self.K = 3.0

    def compute_order_parameter(self) -> Tuple[float, float]:
        """
        Computes Kuramoto order parameter R * e^(i * psi) = (1/N) sum_j e^(i * theta_j).
        R in [0, 1] measures global phase coherence (1.0 = perfect phase-locking).
        psi is the mean global phase angle.
        """
        complex_phases = torch.exp(1j * self.theta.to(torch.complex64))
        mean_phase = torch.mean(complex_phases)
        R = float(torch.abs(mean_phase).item())
        psi = float(torch.angle(mean_phase).item())
        return R, psi

    def inject_perturbation(self, delta_phi: float, gradient_tension: float = 0.2) -> None:
        """
        Injects external/internal perturbation (delta_phi) and field gradient tension.
        Drives the system into POLARIZATION as E_gate charges up.
        """
        delta_tensor = torch.tensor(delta_phi, dtype=torch.float32, device=self.device)
        grad_tensor = torch.tensor(gradient_tension, dtype=torch.float32, device=self.device)

        # Distribute phase perturbation to a random subset of micro-oscillators
        perturb_mask = torch.rand(self.num_oscillators, device=self.device) < 0.3
        self.theta[perturb_mask] = (self.theta[perturb_mask] + delta_tensor) % (2.0 * math.pi)

        # Accumulate field gradient and E_gate barrier pressure
        self.field_gradient += torch.abs(grad_tensor) + torch.abs(delta_tensor) * 0.5
        self.E_gate += torch.abs(grad_tensor) * 1.5 + torch.abs(delta_tensor) * 0.8

        if self.E_gate.item() >= 0.3 * self.threshold_E_gate and self.phase_state == PhaseState.SUPERPOSITION:
            self.phase_state = PhaseState.POLARIZATION

    def step(self, dt: float = 0.05) -> FieldStateSnapshot:
        """
        Advances the field dynamics by dt.
        Applies non-linear Kuramoto coupling, barrier pressure checks, and global phase-locking discharge.
        """
        self.time_step += 1
        R, psi = self.compute_order_parameter()

        # Phase 2 & 3 state transition checks
        if self.E_gate.item() >= self.threshold_E_gate:
            self.phase_state = PhaseState.GLOBAL_PHASE_LOCKING_DISCHARGE

        if self.phase_state == PhaseState.GLOBAL_PHASE_LOCKING_DISCHARGE:
            # Sudden Global Phase-Locking: Force all micro-oscillators toward mean phase psi
            effective_K = self.K * 10.0  # Massive coupling surge
            dtheta = self.omega + effective_K * R * torch.sin(psi - self.theta)
            self.theta = (self.theta + dtheta * dt) % (2.0 * math.pi)

            # E_gate barrier collapses to 0, emitting macro energy burst
            discharge_amount = float(self.E_gate.item()) + float(self.field_gradient.item())
            self.discharged_energy += discharge_amount

            self.E_gate.zero_()
            self.field_gradient.zero_()

            # Transition back to superposition / relaxed state
            self.phase_state = PhaseState.SUPERPOSITION
        else:
            # Standard Kuramoto phase evolution
            effective_K = self.K if self.phase_state == PhaseState.POLARIZATION else (self.K * 0.2)
            dtheta = self.omega + effective_K * R * torch.sin(psi - self.theta)
            self.theta = (self.theta + dtheta * dt) % (2.0 * math.pi)

            # Gradual barrier pressure decay if below threshold
            self.E_gate = torch.clamp(self.E_gate - self.dissipation_rate * dt, min=0.0)
            self.field_gradient = torch.clamp(self.field_gradient - self.dissipation_rate * dt, min=0.0)
            self.discharged_energy = torch.clamp(self.discharged_energy - self.dissipation_rate * 0.1 * dt, min=0.0)

            if self.E_gate.item() < 0.2 * self.threshold_E_gate:
                self.phase_state = PhaseState.SUPERPOSITION

        # Re-compute final order parameter and phase variance
        final_R, _ = self.compute_order_parameter()
        phase_var = float(torch.var(self.theta).item())

        return FieldStateSnapshot(
            scale=self.scale,
            phase_state=self.phase_state,
            num_oscillators=self.num_oscillators,
            order_parameter_R=final_R,
            phase_variance=phase_var,
            barrier_pressure_E_gate=float(self.E_gate.item()),
            field_gradient=float(self.field_gradient.item()),
            discharged_energy=float(self.discharged_energy.item()),
            threshold_E_gate=self.threshold_E_gate,
            time_step=self.time_step
        )

    def reset(self) -> None:
        """Resets engine state to initial uncommitted superposition state."""
        self.theta.uniform_(0.0, 2.0 * math.pi)
        self.E_gate.zero_()
        self.field_gradient.zero_()
        self.discharged_energy.zero_()
        self.phase_state = PhaseState.SUPERPOSITION
        self.time_step = 0
