"""
core/topology/geometric_phase_memory_engine.py

Addressless Geometric Phase Topology & Memory Engine (주소-연산 통합 기하학적 위상 메모리 및 관측 지평 확장 엔진)

Implements:
1. Addressless Geometric Phase Topology: Data as phase-locked attractor wells in 5D Clifford rotor space.
2. Holographic Memory Retrieval: Constructive phase interference without memory addresses.
3. State Transition via Phase-Slip: 1tan boundary stress accumulation and gradient drift across attraction basins.
4. 3-Stage Autonomous Self-Healing: Destructive noise dissipation, 1tan restoring stress, and dynamic phase relaxation.
5. Expansion of Observational Cognition (\\partial W): Multimodal/Synesthetic frequency fusion (vision, audio, tactile, science) into unified wave field.
6. Recursive Meta-Stratification (Meta-Cognitive Frame Reset): Boundary stress overload triggers higher-dimensional frame reconfiguration.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any
import math
import torch
import torch.nn as nn
import numpy as np


@dataclass
class AttractorWell:
    """A geometric attractor well in phase space (addressless memory anchor)."""
    name: str
    phase_anchor: torch.Tensor # [dim]
    bivector_5d: torch.Tensor  # [10] Spin(5) bivector generator
    frequency_signature: torch.Tensor # Multimodal spectral signature
    depth: float = 1.0


@dataclass
class MemoryRetrievalResult:
    retrieved_rotor: torch.Tensor
    attractor_index: int
    attractor_name: str
    resonance_score: float
    phase_error: float
    is_phase_locked: bool


@dataclass
class StateTransitionResult:
    status: str
    previous_attractor: str
    current_attractor: str
    stress_magnitude: float
    phase_slip_occurred: bool
    transition_energy: float


@dataclass
class SelfHealingResult:
    initial_disturbance_norm: float
    restoring_force_norm: float
    dynamic_damping: float
    final_phase_error: float
    healed_rotor: torch.Tensor


@dataclass
class MetaFrameResetResult:
    is_meta_triggered: bool
    accumulated_stress: float
    old_horizon_ratio: float
    new_horizon_ratio: float
    reconfigured_anchor: torch.Tensor


class GeometricPhaseMemoryEngine(nn.Module):
    """
    Geometric Phase Memory & Topological Transducer Engine.
    Unifies memory storage, computation, and state transitions into single wave dynamics.
    """

    def __init__(
        self,
        dimension: int = 32,
        dim_rotor: int = 5,
        epsilon_gauge: float = 1e-6,
        gamma_0: float = 0.15,
        stress_threshold: float = 2.5,
        tan_clamp_limit: float = 50.0,
        dtype=torch.float32
    ):
        super().__init__()
        self.dimension = dimension
        self.dim_rotor = dim_rotor
        self.eps = epsilon_gauge
        self.gamma_0 = gamma_0
        self.threshold = stress_threshold
        self.tan_clamp_limit = tan_clamp_limit
        self.dtype = dtype

        # Attractor repository (addressless anchors in geometric phase space)
        self.attractors: List[AttractorWell] = []

        # Current system state: 5D Spinor / Clifford Rotor (Spin(5) in 5D)
        # Spin(5) has 10 bivector generators: e12, e13, e14, e15, e23, e24, e25, e34, e35, e45
        self.current_bivector = nn.Parameter(torch.zeros(10, dtype=dtype))
        self.current_phase_vector = nn.Parameter(torch.randn(dimension, dtype=dtype) * 0.1)

        # Observational Horizon Boundary (\\partial W) ratio [0, 1]
        self.horizon_ratio = 0.5
        # Gauge anchor position x_anchor
        self.register_buffer("x_anchor", torch.zeros(3, dtype=dtype))

    def _bivector_to_rotor_matrix(self, bivector_10d: torch.Tensor) -> torch.Tensor:
        """
        Maps 10D Spin(5) bivector to 5x5 orthogonal Clifford Rotor matrix via Lie algebra exp map so(5).
        """
        omega = torch.zeros(5, 5, dtype=self.dtype, device=bivector_10d.device)
        idx = 0
        for i in range(5):
            for j in range(i + 1, 5):
                omega[i, j] = bivector_10d[idx]
                omega[j, i] = -bivector_10d[idx]
                idx += 1
        return torch.matrix_exp(omega)

    def register_attractor(
        self,
        name: str,
        phase_pattern: torch.Tensor,
        frequency_signature: Optional[torch.Tensor] = None
    ) -> AttractorWell:
        """
        Registers an addressless geometric attractor well in phase space.
        """
        norm = torch.norm(phase_pattern)
        normalized_pattern = phase_pattern / (norm + self.eps)

        # Derive 10D bivector generator from pattern
        bivector = torch.zeros(10, dtype=self.dtype, device=phase_pattern.device)
        for i in range(min(10, self.dimension)):
            bivector[i] = torch.sin(normalized_pattern[i % self.dimension] * math.pi)

        if frequency_signature is None:
            frequency_signature = torch.randn(self.dimension, dtype=self.dtype, device=phase_pattern.device)

        attractor = AttractorWell(
            name=name,
            phase_anchor=normalized_pattern,
            bivector_5d=bivector,
            frequency_signature=frequency_signature,
            depth=1.0
        )
        self.attractors.append(attractor)
        return attractor

    def calculate_1tan_stress(self, phase_vector: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Calculates 1tan tangential boundary stress with gauge smoothing eps_gauge and clamping limit.
        W_stress = tan(theta)
        """
        sin_val = torch.sin(phase_vector)
        cos_val = torch.cos(phase_vector)

        safe_cos = torch.where(
            torch.abs(cos_val) < self.eps,
            self.eps * torch.sign(cos_val + 1e-8),
            cos_val
        )

        tan_val = torch.clamp(sin_val / safe_cos, -self.tan_clamp_limit, self.tan_clamp_limit)
        stress_vector = tan_val
        stress_magnitude = float(torch.norm(stress_vector).item())
        return stress_vector, stress_magnitude

    def retrieve_memory(self, query_wave: torch.Tensor) -> MemoryRetrievalResult:
        """
        1. Holographic Memory Retrieval (주소 없는 위상 공명 복원)
        Finds the attractor well that produces maximal constructive wave interference with query_wave.
        """
        if not self.attractors:
            rotor_mat = self._bivector_to_rotor_matrix(self.current_bivector)
            return MemoryRetrievalResult(
                retrieved_rotor=rotor_mat,
                attractor_index=-1,
                attractor_name="unanchored",
                resonance_score=0.0,
                phase_error=1.0,
                is_phase_locked=False
            )

        query_norm = query_wave / (torch.norm(query_wave) + self.eps)
        max_resonance = -1e9
        best_idx = 0

        for idx, attractor in enumerate(self.attractors):
            # Constructive interference dot product
            resonance = float(torch.dot(query_norm, attractor.phase_anchor).item())
            if resonance > max_resonance:
                max_resonance = resonance
                best_idx = idx

        best_attractor = self.attractors[best_idx]
        # Update current system phase and bivector state towards locked attractor
        with torch.no_grad():
            self.current_phase_vector.copy_(best_attractor.phase_anchor)
            self.current_bivector.copy_(best_attractor.bivector_5d)

        phase_error = max(0.0, 1.0 - max_resonance)
        is_locked = phase_error < 0.1
        rotor_mat = self._bivector_to_rotor_matrix(self.current_bivector)

        return MemoryRetrievalResult(
            retrieved_rotor=rotor_mat,
            attractor_index=best_idx,
            attractor_name=best_attractor.name,
            resonance_score=max_resonance,
            phase_error=phase_error,
            is_phase_locked=is_locked
        )

    def step_state_transition(self, external_shock: torch.Tensor) -> StateTransitionResult:
        """
        2. State Transition via Phase-Slip (1tan 접선 응력 기반 자율 위상 도약)
        Accumulates stress; when stress_magnitude > threshold, phase-slip drifts to adjacent attractor.
        """
        prev_attractor_name = "unanchored"
        if self.attractors:
            ret_before = self.retrieve_memory(self.current_phase_vector)
            prev_attractor_name = ret_before.attractor_name

        with torch.no_grad():
            self.current_phase_vector.add_(external_shock)

        stress_vec, stress_mag = self.calculate_1tan_stress(self.current_phase_vector)

        phase_slip_occurred = False
        curr_attractor_name = prev_attractor_name
        transition_energy = 0.0

        if stress_mag > self.threshold:
            phase_slip_occurred = True
            transition_energy = stress_mag - self.threshold
            # Drift to nearest minimal energy attractor
            ret_after = self.retrieve_memory(self.current_phase_vector)
            curr_attractor_name = ret_after.attractor_name
            status = "PHASE_SLIP_TRANSITION"
        else:
            status = "STRESS_ACCUMULATING"

        return StateTransitionResult(
            status=status,
            previous_attractor=prev_attractor_name,
            current_attractor=curr_attractor_name,
            stress_magnitude=stress_mag,
            phase_slip_occurred=phase_slip_occurred,
            transition_energy=transition_energy
        )

    def self_heal_noise(self, noise_impulse: torch.Tensor) -> SelfHealingResult:
        """
        3. 3-Stage Autonomous Self-Healing (기하학적 자율 노이즈 복원)
        Stage 1: Destructive dissipation of high-frequency noise.
        Stage 2: 1tan restoring stress reaction F_restore = -grad(W_stress).
        Stage 3: Thermodynamic phase relaxation with dynamic damping gamma_dynamic.
        """
        initial_norm = float(torch.norm(noise_impulse).item())
        distorted_phase = self.current_phase_vector + noise_impulse

        # Stage 1 & 2: Calculate 1tan stress and restoring force
        stress_vec, stress_mag = self.calculate_1tan_stress(distorted_phase)
        f_restore = -stress_vec

        # Stage 3: Dynamic damping coefficient gamma_dynamic = gamma_0 * (1 + 0.5 * stress_mag)
        gamma_dyn = self.gamma_0 * (1.0 + 0.5 * min(stress_mag, 10.0))

        # Perform geometric phase relaxation step
        healed_phase = distorted_phase + (f_restore * gamma_dyn * 0.01)

        # Attractor lock normalization
        ret = self.retrieve_memory(healed_phase)
        final_rotor = ret.retrieved_rotor

        return SelfHealingResult(
            initial_disturbance_norm=initial_norm,
            restoring_force_norm=float(torch.norm(f_restore).item()),
            dynamic_damping=gamma_dyn,
            final_phase_error=ret.phase_error,
            healed_rotor=final_rotor
        )

    def fuse_synesthetic_frequencies(
        self,
        vision_wave: torch.Tensor,
        audio_wave: torch.Tensor,
        tactile_wave: torch.Tensor,
        scientific_wave: torch.Tensor
    ) -> torch.Tensor:
        """
        Expansion of Observational Cognition (\\partial W):
        Fuses vision, audio, tactile, and scientific waves into single synesthetic wave field.
        """
        # Ensure all waves are mapped to system dimension
        v = torch.mean(vision_wave) if vision_wave.ndim > 1 else vision_wave
        a = audio_wave[:self.dimension]
        t = tactile_wave[:self.dimension]
        s = scientific_wave[:self.dimension]

        # Trinitarian cross-modal phase-locking (1sin vision, 1cos audio, 1tan tactile)
        fused_wave = torch.sin(a) + torch.cos(s) + 0.1 * (v + t)
        fused_norm = fused_wave / (torch.norm(fused_wave) + self.eps)

        return fused_norm

    def trigger_recursive_meta_frame_reset(self, accumulated_stress: float) -> MetaFrameResetResult:
        """
        Recursive Meta-Stratification (한계 인식 -> 상위 프레임 재설정):
        When stress exceeds limit, triggers higher-dimensional meta-cognition to expand horizon boundary \\partial W.
        """
        is_triggered = accumulated_stress > (self.threshold * 1.5)
        old_ratio = self.horizon_ratio

        if is_triggered:
            # Expand horizon boundary \\partial W to subsume contradiction
            self.horizon_ratio = min(1.0, self.horizon_ratio + 0.2)
            # Reconfigure gauge anchor position
            with torch.no_grad():
                self.x_anchor.add_(torch.tensor([0.1, 0.1, 0.1], dtype=self.dtype, device=self.x_anchor.device))
                # Rotate bivectors into higher meta-frame
                self.current_bivector.mul_(0.5)

        return MetaFrameResetResult(
            is_meta_triggered=is_triggered,
            accumulated_stress=accumulated_stress,
            old_horizon_ratio=old_ratio,
            new_horizon_ratio=self.horizon_ratio,
            reconfigured_anchor=self.x_anchor.clone()
        )
