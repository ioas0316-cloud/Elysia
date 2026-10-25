"""
Multi-Scale Wave Sensorimotor Engine (다중 스케일 감각-위상 및 감각-운동 엔진)

Implements the 3-scale sensory-phase ecosystem and closed-loop enactive cognition:
1. Micro-Scale (Alpha-Helical Phase Prism Gate):
   - Helical geometry matching for molecular/chemical/token frequencies (e.g. Na+ ion pocket, scent, taste, micro atomic waves).
   - Dynamic gate barrier energy (E_gate) computation and helical phase matching filter.

2. Meso-Scale (Cellular/Tissue Scale-Bridging PLL):
   - Multi-scale brainwave frequency hierarchy (Delta 0.5-4Hz, Theta 4-8Hz, Alpha 8-12Hz, Beta 12-30Hz, Gamma 30-100Hz).
   - Amplifies and synchronizes micro phase perturbations into macro cognitive resonance.

3. Macro-Scale (Macro Sensory Prism Transducer):
   - Translates external physical wave spectra (optical light spectra, acoustic vibrations, chemical/taste resonance, tactile shear/normal stress, environmental spatial field) into internal 5D phase perturbations.

4. Sensorimotor Closed-Loop & Redirection (Closed-Loop Enactive Cognition):
   - Detects phase dead-ends (E_gate > threshold) and triggers rotor angular phase shifts (Delta phi) for rerouting.
   - Closed-loop action loop executing phase alignment to relax barrier tension (E_gate -> 0).
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import math
import torch
import torch.nn as nn
import numpy as np


@dataclass
class MicroHelicalGateOutput:
    helical_phase: torch.Tensor
    gate_energy: float
    passed: bool
    affinity_score: float


@dataclass
class MesoBandState:
    delta_phase: torch.Tensor  # 0.5 - 4 Hz
    theta_phase: torch.Tensor  # 4 - 8 Hz
    alpha_phase: torch.Tensor  # 8 - 12 Hz
    beta_phase: torch.Tensor   # 12 - 30 Hz
    gamma_phase: torch.Tensor  # 30 - 100 Hz
    global_coherence: float


@dataclass
class MacroSensoryOutput:
    optical_phase: torch.Tensor
    acoustic_phase: torch.Tensor
    chemical_phase: torch.Tensor
    tactile_phase: torch.Tensor
    spatial_phase: torch.Tensor
    unified_5d_perturbation: torch.Tensor


@dataclass
class SensorimotorLoopOutput:
    micro_output: MicroHelicalGateOutput
    meso_state: MesoBandState
    macro_sensory: MacroSensoryOutput
    current_e_gate: float
    is_dead_end: bool
    redirected_phase_shift: float
    action_vector: torch.Tensor
    relaxed_e_gate: float
    closed_loop_resonance: float


class HelicalPhasePrismGate(nn.Module):
    """
    Micro-Scale: Alpha-Helical Phase Prism Gate.
    Models micro-molecular/chemical/token helical pockets.
    """

    def __init__(self, dimension: int = 64, helical_pitch: float = 2.0 * math.pi / 3.6):
        super().__init__()
        self.dimension = dimension
        self.helical_pitch = helical_pitch
        # Helical projection matrix
        self.helical_transform = nn.Linear(dimension, dimension, bias=False)
        nn.init.orthogonal_(self.helical_transform.weight)

    def forward(
        self,
        input_wave: torch.Tensor,
        target_receptor_pocket: Optional[torch.Tensor] = None,
        rotor_phi: Optional[torch.Tensor] = None
    ) -> MicroHelicalGateOutput:
        device = input_wave.device
        if input_wave.dim() == 1:
            input_wave = input_wave.unsqueeze(0)

        # Helical phase rotation along 3D alpha-helix trajectory
        hel_transformed = self.helical_transform(input_wave)
        helical_phase = torch.atan2(hel_transformed, input_wave + 1e-8)

        if rotor_phi is not None:
            helical_phase = helical_phase + rotor_phi

        if target_receptor_pocket is None:
            target_receptor_pocket = torch.zeros_like(input_wave)
        elif target_receptor_pocket.dim() == 1:
            target_receptor_pocket = target_receptor_pocket.unsqueeze(0)

        # Angular phase difference wrapped in [-pi, pi]
        phase_mismatch = torch.remainder(helical_phase - target_receptor_pocket + math.pi, 2.0 * math.pi) - math.pi
        e_gate = float(torch.mean(phase_mismatch ** 2).item())

        affinity_score = float(torch.exp(-torch.abs(phase_mismatch)).mean().item())
        passed = e_gate < 1.0  # Gate threshold for micro-helix entry

        return MicroHelicalGateOutput(
            helical_phase=helical_phase.squeeze(0),
            gate_energy=e_gate,
            passed=passed,
            affinity_score=affinity_score
        )


class MesoScaleBridgingPLL(nn.Module):
    """
    Meso-Scale: Scale-Bridging Phase-Locked Loop (PLL).
    Connects micro phase perturbations to multi-band brainwave rhythms.
    """

    def __init__(self, dimension: int = 64, coupling_k: float = 0.8):
        super().__init__()
        self.dimension = dimension
        self.coupling_k = coupling_k

        # Frequency band natural frequencies (normalized)
        self.freq_delta = 2.0 / 100.0   # Delta (0.5-4 Hz)
        self.freq_theta = 6.0 / 100.0   # Theta (4-8 Hz)
        self.freq_alpha = 10.0 / 100.0  # Alpha (8-12 Hz)
        self.freq_beta = 20.0 / 100.0   # Beta (12-30 Hz)
        self.freq_gamma = 40.0 / 100.0  # Gamma (30-100 Hz)

        # Internal oscillator phase buffers
        self.register_buffer("phase_delta", torch.rand(dimension) * 2.0 * math.pi)
        self.register_buffer("phase_theta", torch.rand(dimension) * 2.0 * math.pi)
        self.register_buffer("phase_alpha", torch.rand(dimension) * 2.0 * math.pi)
        self.register_buffer("phase_beta", torch.rand(dimension) * 2.0 * math.pi)
        self.register_buffer("phase_gamma", torch.rand(dimension) * 2.0 * math.pi)

    def step_synchronization(self, micro_perturbation: torch.Tensor, dt: float = 0.1) -> MesoBandState:
        if micro_perturbation.dim() > 1:
            micro_perturbation = micro_perturbation.mean(dim=0)

        # Update oscillator phases using Kuramoto-style phase locking
        d_phase = micro_perturbation - self.phase_gamma
        self.phase_gamma = (self.phase_gamma + 2.0 * math.pi * self.freq_gamma * dt + self.coupling_k * torch.sin(d_phase)) % (2.0 * math.pi)

        d_beta = self.phase_gamma - self.phase_beta
        self.phase_beta = (self.phase_beta + 2.0 * math.pi * self.freq_beta * dt + self.coupling_k * torch.sin(d_beta)) % (2.0 * math.pi)

        d_alpha = self.phase_beta - self.phase_alpha
        self.phase_alpha = (self.phase_alpha + 2.0 * math.pi * self.freq_alpha * dt + self.coupling_k * torch.sin(d_alpha)) % (2.0 * math.pi)

        d_theta = self.phase_alpha - self.phase_theta
        self.phase_theta = (self.phase_theta + 2.0 * math.pi * self.freq_theta * dt + self.coupling_k * torch.sin(d_theta)) % (2.0 * math.pi)

        d_delta = self.phase_theta - self.phase_delta
        self.phase_delta = (self.phase_delta + 2.0 * math.pi * self.freq_delta * dt + self.coupling_k * torch.sin(d_delta)) % (2.0 * math.pi)

        # Global Kuramoto order parameter (coherence)
        all_phases = torch.stack([
            self.phase_delta, self.phase_theta, self.phase_alpha,
            self.phase_beta, self.phase_gamma
        ], dim=0)
        complex_order = torch.mean(torch.exp(1j * all_phases))
        global_coherence = float(torch.abs(complex_order).item())

        return MesoBandState(
            delta_phase=self.phase_delta.clone(),
            theta_phase=self.phase_theta.clone(),
            alpha_phase=self.phase_alpha.clone(),
            beta_phase=self.phase_beta.clone(),
            gamma_phase=self.phase_gamma.clone(),
            global_coherence=global_coherence
        )


class MacroSensoryPrismTransducer(nn.Module):
    """
    Macro-Scale: Macro Sensory Prism Transducer.
    Translates physical wave spectra into unified 5D internal phase perturbations.
    Supports optical, acoustic, chemical/ion, tactile, and spatial field modalities.
    """

    def __init__(self, dimension: int = 64):
        super().__init__()
        self.dimension = dimension
        # Modality 5D projection mapping
        self.proj_5d = nn.Linear(dimension, 5, bias=False)

    def forward(
        self,
        optical_spectrum: Optional[torch.Tensor] = None,
        acoustic_wave: Optional[torch.Tensor] = None,
        chemical_ion_resonance: Optional[torch.Tensor] = None,
        tactile_stress: Optional[torch.Tensor] = None,
        spatial_gradient: Optional[torch.Tensor] = None
    ) -> MacroSensoryOutput:
        device = self.proj_5d.weight.device

        def process_modality(mod_tensor: Optional[torch.Tensor]) -> torch.Tensor:
            if mod_tensor is None:
                return torch.zeros(self.dimension, device=device)
            if mod_tensor.dim() > 1:
                mod_tensor = mod_tensor.squeeze()
            if mod_tensor.numel() != self.dimension:
                mod_tensor = nn.functional.adaptive_avg_pool1d(mod_tensor.view(1, 1, -1), self.dimension).squeeze()
            return torch.atan2(mod_tensor, torch.roll(mod_tensor, shifts=1) + 1e-8)

        opt_phase = process_modality(optical_spectrum)
        acou_phase = process_modality(acoustic_wave)
        chem_phase = process_modality(chemical_ion_resonance)
        tact_phase = process_modality(tactile_stress)
        spat_phase = process_modality(spatial_gradient)

        # Superposition of all active sensory phase components
        stacked = torch.stack([opt_phase, acou_phase, chem_phase, tact_phase, spat_phase], dim=0)
        unified_wave = stacked.mean(dim=0)

        # Map to 5D unified phase perturbation vector
        unified_5d = torch.tanh(self.proj_5d(unified_wave))

        return MacroSensoryOutput(
            optical_phase=opt_phase,
            acoustic_phase=acou_phase,
            chemical_phase=chem_phase,
            tactile_phase=tact_phase,
            spatial_phase=spat_phase,
            unified_5d_perturbation=unified_5d
        )


class MultiScaleWaveSensorimotorEngine(nn.Module):
    """
    Unified Multi-Scale Wave Sensorimotor Engine.
    Combines Micro-Helical Prism, Meso-PLL Scale-Bridging, Macro-Sensory Prism,
    and Closed-Loop Sensorimotor Redirection & Energy Relaxation.
    """

    def __init__(
        self,
        dimension: int = 64,
        dead_end_threshold: float = 1.5,
        learning_rate: float = 0.1
    ):
        super().__init__()
        self.dimension = dimension
        self.dead_end_threshold = dead_end_threshold
        self.learning_rate = learning_rate

        self.micro_gate = HelicalPhasePrismGate(dimension=dimension)
        self.meso_pll = MesoScaleBridgingPLL(dimension=dimension)
        self.macro_transducer = MacroSensoryPrismTransducer(dimension=dimension)

        # Internal Rotor Angle Offset (phi_rotor)
        self.register_buffer("rotor_phi", torch.tensor(0.0))
        # Motor Action Generator
        self.action_generator = nn.Linear(5, dimension, bias=False)

    def forward(
        self,
        optical_spectrum: Optional[torch.Tensor] = None,
        acoustic_wave: Optional[torch.Tensor] = None,
        chemical_ion_resonance: Optional[torch.Tensor] = None,
        tactile_stress: Optional[torch.Tensor] = None,
        spatial_gradient: Optional[torch.Tensor] = None,
        target_receptor_pocket: Optional[torch.Tensor] = None,
        dt: float = 0.1
    ) -> SensorimotorLoopOutput:
        # 1. Macro-Scale Wave Transduction
        macro_sensory = self.macro_transducer(
            optical_spectrum=optical_spectrum,
            acoustic_wave=acoustic_wave,
            chemical_ion_resonance=chemical_ion_resonance,
            tactile_stress=tactile_stress,
            spatial_gradient=spatial_gradient
        )

        # 2. Micro-Scale Helical Gate Analysis
        micro_input = macro_sensory.optical_phase + macro_sensory.chemical_phase + macro_sensory.tactile_phase
        micro_out = self.micro_gate(micro_input, target_receptor_pocket=target_receptor_pocket, rotor_phi=self.rotor_phi)

        # 3. Meso-Scale PLL Synchronization
        meso_state = self.meso_pll.step_synchronization(micro_out.helical_phase, dt=dt)

        # 4. Check Phase Dead-End (E_gate > threshold)
        is_dead_end = micro_out.gate_energy > self.dead_end_threshold
        redirected_phase_shift = 0.0

        if is_dead_end:
            # Dead-end detected: Shift rotor angle to bypass topological barrier
            redirected_phase_shift = math.pi / 4.0  # 45-degree helical reroute
            self.rotor_phi += redirected_phase_shift

        # 5. Closed-Loop Sensorimotor Action (Phase Alignment to relax E_gate -> 0)
        aligned_5d = macro_sensory.unified_5d_perturbation * torch.cos(self.rotor_phi)
        action_vector = torch.tanh(self.action_generator(aligned_5d))

        action_magnitude = float(torch.norm(action_vector).item())
        # Apply motor action feedback to align rotor_phi towards barrier relaxation
        self.rotor_phi -= self.learning_rate * action_magnitude * 0.1

        relaxed_e_gate = max(0.0, micro_out.gate_energy - action_magnitude * self.learning_rate)

        # Resonance score combining coherence and barrier relaxation
        closed_loop_resonance = meso_state.global_coherence * (1.0 / (1.0 + relaxed_e_gate))

        return SensorimotorLoopOutput(
            micro_output=micro_out,
            meso_state=meso_state,
            macro_sensory=macro_sensory,
            current_e_gate=micro_out.gate_energy,
            is_dead_end=is_dead_end,
            redirected_phase_shift=redirected_phase_shift,
            action_vector=action_vector,
            relaxed_e_gate=relaxed_e_gate,
            closed_loop_resonance=closed_loop_resonance
        )
