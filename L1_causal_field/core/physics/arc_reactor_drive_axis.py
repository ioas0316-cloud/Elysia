"""
[Arc Reactor Drive Axis Engine (아크원자로 국소적 제로 구동축 엔진)]

Core Philosophy:
"에너지를 억지로 생성하려고 애쓸 필요 없이, 자연의 파동과 스펙트럼이 가진
가속도와 장력의 기울기 위에 '저항이 소거된 초전도적 관성 궤도(Local Zero Domain)'를 세운다."

Key Mechanisms:
1. Inward Tension & Equilibrium Field (수렴 장력과 동적 평형):
   - 팽창하려는 엔트로피/에너지에 맞서 중심 코어로 수렴하는 필드 장력(F_inward).
2. Intent Boundary & Local Zero Transition (의도의 테두리와 국소적 제로 전이):
   - 설정된 의도(Intent) 영역 내부에서 내부 저항 R -> 0으로 수렴하는 초전도 제로존.
3. Phase Acceleration & Flow Dynamics (위상 가속 및 기울기 구동):
   - 위상 기울기(Potential Gradient)를 마찰 없이 순수 회전력(Phase Velocity ω)으로 치환.
4. Inertial Flywheel Conservation (관성 플라이휠 회전력 유지):
   - 외부 자극 및 에너지가 소멸하더라도 보존되는 고유 회전 주기와 자립적 구조력.
5. Discerning Tension & Noise-to-Melody Transformation (분별적 장력과 노이즈-음률 승화):
   - 외부의 무질서한 노이즈 파형을 시스템의 위상축과 공명시켜 유의미한 화음(Melody)으로 재정렬.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from core.physics.causal_field import CausalField, InformationVoxel


@dataclass
class LocalZeroFieldState:
    """Represents the current state metrics of the Local Zero Drive Axis."""
    inward_tension: float = 0.0          # Strength of inward tension pulling towards core
    internal_resistance: float = 1.0     # Internal friction / resistance R (R -> 0 in zero zone)
    phase_velocity: float = 0.0          # Rotational / phase angular velocity (rad/s)
    flywheel_momentum: float = 0.0       # Preserved inertial momentum L = I * omega
    resonant_harmony_score: float = 0.0  # Quality score of noise transformed into melody [0, 1]
    energy_efficiency: float = 0.0       # Superconducting efficiency (1 - R/R_max)
    active_domain_radius: float = 2.0    # Local Zero boundary radius
    core_center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


class ArcReactorDriveEngine:
    """
    [Arc Reactor Resonant Drive Axis Engine]

    Implements the Drive Axis governing continuous energy fields, local zero resistance,
    and noise-to-melody phase alignment in the Elysia framework.
    """
    def __init__(
        self,
        dimensions: int = 3,
        core_center: Optional[np.ndarray] = None,
        domain_radius: float = 2.0,
        flywheel_inertia: float = 10.0,
        base_resistance: float = 1.0,
        intent_phase_frequency: float = 1.0
    ):
        self.dimensions = dimensions
        self.core_center = (
            np.array(core_center, dtype=np.float32)
            if core_center is not None
            else np.zeros(dimensions, dtype=np.float32)
        )
        self.domain_radius = float(domain_radius)
        self.flywheel_inertia = float(flywheel_inertia)
        self.base_resistance = float(base_resistance)
        self.intent_phase_frequency = float(intent_phase_frequency)

        # Dynamic State Variables
        self.phase_angle: float = 0.0             # Rotor phase angle theta in [0, 2pi)
        self.phase_velocity: float = 0.0          # Angular speed omega
        self.current_resistance: float = base_resistance
        self.inward_tension_mag: float = 0.0
        self.resonant_harmony: float = 0.0
        self.time_elapsed: float = 0.0

        # Accumulated harmonic melody spectrum
        self.harmonic_spectrum: List[float] = []

    def compute_local_resistance(self, distance_from_core: float) -> float:
        """
        Calculates local resistance R(r) as a continuous function of distance r from core center.
        Inside the Local Zero domain (r <= domain_radius), resistance drops exponentially/superconductively to 0.

        R(r) = R_0 * tanh((r / r_zero)^4)
        """
        ratio = distance_from_core / (self.domain_radius + 1e-9)
        # Continuous smooth transition: 0 inside domain, base_resistance outside
        resistance = self.base_resistance * np.tanh(ratio ** 4)
        return float(resistance)

    def compute_inward_tension(self, position: np.ndarray, mass: float = 1.0) -> np.ndarray:
        """
        Computes the inward tension force vector pulling towards the core center.
        Balancing outward entropy expansion with strong core attraction.

        F_tension = - k * mass * (pos - pos_core) * exp(-||pos - pos_core||^2 / (2 * sigma^2))
        """
        pos = np.array(position, dtype=np.float32)
        diff = pos - self.core_center
        dist = np.linalg.norm(diff)

        if dist < 1e-9:
            return np.zeros(self.dimensions, dtype=np.float32)

        # Inward Gaussian tension force profile
        sigma = self.domain_radius * 1.5
        k_tension = 5.0
        force_mag = k_tension * mass * dist * np.exp(-(dist ** 2) / (2.0 * sigma ** 2))

        # Vector direction pointing INWARD toward core center
        inward_dir = -diff / dist
        return inward_dir * force_mag

    def process_discerning_tension(
        self,
        raw_noise_signal: np.ndarray,
        sample_rate: float = 100.0
    ) -> Tuple[np.ndarray, float]:
        """
        [Discerning Tension: Turning Noise into Resonant Melody]

        Rather than filtering out ambient noise by brute-force masking,
        discerning tension aligns incoming unrefined wave components with the system's
        intent phase frequency.

        Components matching or harmonizing with intent frequency lock into phase (Melody),
        while unaligned phase noise is absorbed into the potential gradient.

        Returns:
            harmonized_signal: Clean resonant wave output.
            harmony_score: Coherence ratio in [0.0, 1.0].
        """
        raw = np.array(raw_noise_signal, dtype=np.float32).flatten()
        if len(raw) == 0:
            return np.zeros(0, dtype=np.float32), 1.0

        n = len(raw)
        t = np.arange(n, dtype=np.float32) / sample_rate

        # Intent reference wave: sin(2pi * f_intent * t + theta)
        intent_wave = np.sin(2.0 * np.pi * self.intent_phase_frequency * t + self.phase_angle)
        harmonic_2nd = 0.5 * np.sin(4.0 * np.pi * self.intent_phase_frequency * t + self.phase_angle)
        harmonic_3rd = 0.25 * np.sin(6.0 * np.pi * self.intent_phase_frequency * t + self.phase_angle)

        reference_melody = intent_wave + harmonic_2nd + harmonic_3rd
        ref_norm = np.linalg.norm(reference_melody) + 1e-9

        # Project raw noise onto intent reference melody space
        coherence_projection = np.dot(raw, reference_melody) / ref_norm

        # Discerning phase locking: aligned components amplify reference melody
        phase_locked_melody = coherence_projection * (reference_melody / ref_norm)

        # Calculate harmony score (ratio of energy in harmonic phase vs total raw energy)
        raw_energy = np.sum(raw ** 2) + 1e-9
        melody_energy = np.sum(phase_locked_melody ** 2)
        harmony_score = float(np.clip(melody_energy / raw_energy, 0.0, 1.0))

        # Update engine's resonant harmony metric
        self.resonant_harmony = 0.8 * self.resonant_harmony + 0.2 * harmony_score
        self.harmonic_spectrum.append(float(coherence_projection))

        return phase_locked_melody, float(harmony_score)

    def step(
        self,
        dt: float = 0.1,
        external_potential_gradient: float = 0.0,
        distance_to_core: float = 0.0,
        raw_noise_signal: Optional[np.ndarray] = None
    ) -> LocalZeroFieldState:
        """
        Advances the Arc Reactor Drive Axis dynamics by dt.

        1. Evaluates local resistance R(r) based on distance to core center.
        2. Applies phase acceleration driven by external potential gradient:
           I * (d_omega / dt) = Torque_gradient - R * omega
        3. Integrates phase angle theta = theta + omega * dt.
        4. Calculates flywheel momentum L = I * omega.
        5. Processes discerning tension if noise signal is provided.
        """
        self.time_elapsed += dt

        # 1. Local Zero resistance evaluation
        self.current_resistance = self.compute_local_resistance(distance_to_core)

        # 2. Torque and Phase Acceleration
        # Gradient accelerates rotation; internal resistance damps rotation.
        # In the Local Zero domain (R -> 0), damping vanishes -> infinite/lossless acceleration.
        torque_input = float(external_potential_gradient)
        damping_torque = self.current_resistance * self.phase_velocity

        angular_acceleration = (torque_input - damping_torque) / self.flywheel_inertia
        self.phase_velocity += angular_acceleration * dt

        # Prevent negative velocity if damping overwhelms without torque
        if self.phase_velocity < 0.0 and torque_input == 0.0:
            self.phase_velocity = 0.0

        # 3. Phase Angle integration
        self.phase_angle = (self.phase_angle + self.phase_velocity * dt) % (2.0 * np.pi)

        # 4. Flywheel Momentum L = I * omega
        flywheel_momentum = self.flywheel_inertia * self.phase_velocity

        # 5. Noise Discerning Process if signal provided
        if raw_noise_signal is not None:
            _, harmony_score = self.process_discerning_tension(raw_noise_signal)
        else:
            harmony_score = self.resonant_harmony

        # Compute Inward Tension magnitude at distance_to_core
        dummy_pos = self.core_center + np.array([distance_to_core] + [0.0] * (self.dimensions - 1), dtype=np.float32)
        tension_vec = self.compute_inward_tension(dummy_pos)
        self.inward_tension_mag = float(np.linalg.norm(tension_vec))

        # Efficiency: 1.0 when resistance is 0
        efficiency = float(max(0.0, 1.0 - (self.current_resistance / self.base_resistance)))

        return LocalZeroFieldState(
            inward_tension=self.inward_tension_mag,
            internal_resistance=self.current_resistance,
            phase_velocity=self.phase_velocity,
            flywheel_momentum=flywheel_momentum,
            resonant_harmony_score=harmony_score,
            energy_efficiency=efficiency,
            active_domain_radius=self.domain_radius,
            core_center=self.core_center.copy()
        )

    def couple_causal_field(self, causal_field: CausalField, dt: float = 0.1):
        """
        [CausalField Coupling]
        Applies Arc Reactor Drive Axis physics to all InformationVoxels in a CausalField:
        - Applies inward tension force toward the core center.
        - Inside the Local Zero domain, reduces voxel friction and sets superconducting conductance.
        - Synchronizes voxel rotor phase angle with the drive axis phase_angle.
        """
        for voxel in causal_field.voxels.values():
            dist_to_core = float(np.linalg.norm(voxel.position - self.core_center))

            # 1. Apply inward tension force
            tension_force = self.compute_inward_tension(voxel.position, mass=voxel.mass)
            voxel.velocity += (tension_force / voxel.mass) * dt

            # 2. Local Zero superconducting state inside domain
            local_r = self.compute_local_resistance(dist_to_core)
            if local_r < self.base_resistance * 0.1: # Inside Local Zero domain
                # Conductance = 1 / R -> infinite / maximum conductance
                voxel.conductance = 100.0
                # Minimal damping inside zero zone
                damping = 1.0 - (local_r * 0.01 * dt)
                voxel.velocity *= damping
            else:
                voxel.conductance = 1.0 / (local_r + 1e-5)

            # 3. Synchronize Phase Angle
            voxel.phase_angle = self.phase_angle

    def trigger_dynamic_topology_expansion(
        self,
        causal_field: CausalField,
        harmonic_signature: np.ndarray,
        harmony_score: float,
        threshold: float = 0.6
    ) -> Optional[InformationVoxel]:
        """
        [Dynamic Topology Expansion (동적 토폴로지 구조적 자율 확장)]

        When a new resonant harmonic wave is extracted (harmony_score >= threshold),
        the system checks if existing voxels in the CausalField can absorb it.
        If no existing voxel matches the harmonic signature, it creates a new non-colliding
        knowledge node/voxel in a new topological dimension/offset, binding it with
        ConnectivityBeams to neighboring nodes without overwriting existing memory.
        """
        if harmony_score < threshold:
            return None

        sig = np.array(harmonic_signature, dtype=np.float32).flatten()
        sig_dim = len(sig)

        # Check existing voxels for phase/tensor alignment
        best_match_score = 0.0
        best_voxel = None

        for voxel in causal_field.voxels.values():
            v_tensor = voxel.tensor.flatten()
            min_dim = min(len(v_tensor), sig_dim)
            if min_dim > 0:
                dot_val = np.dot(v_tensor[:min_dim], sig[:min_dim])
                norm_prod = (np.linalg.norm(v_tensor[:min_dim]) * np.linalg.norm(sig[:min_dim])) + 1e-9
                alignment = float(dot_val / norm_prod)
                if alignment > best_match_score:
                    best_match_score = alignment
                    best_voxel = voxel

        # If existing node resonates well enough (> 0.85 alignment), update it smoothly
        if best_match_score > 0.85 and best_voxel is not None:
            best_voxel.potential += harmony_score * 0.1
            return best_voxel

        # Otherwise, spawn a NEW non-colliding higher-dimensional knowledge voxel
        node_count = len(causal_field.voxels)
        new_id = f"expanded_node_{node_count + 1}"

        # Position offset along orthogonal dimension / phase angle
        phase_offset = np.array([
            np.cos(self.phase_angle) * (self.domain_radius + 0.5),
            np.sin(self.phase_angle) * (self.domain_radius + 0.5),
            (node_count + 1) * 0.2
        ], dtype=np.float32)

        if len(phase_offset) < causal_field.dimensions:
            phase_offset = np.pad(phase_offset, (0, causal_field.dimensions - len(phase_offset)))
        elif len(phase_offset) > causal_field.dimensions:
            phase_offset = phase_offset[:causal_field.dimensions]

        new_voxel = InformationVoxel(
            id=new_id,
            content=f"Expanded Resonant Concept Cluster #{node_count + 1}",
            tensor=sig.copy(),
            mass=1.0 + harmony_score,
            position=self.core_center + phase_offset,
            phase_angle=self.phase_angle,
            conductance=100.0 # High superconducting conductance
        )
        causal_field.add_voxel(new_voxel)

        # Link to nearest existing voxel to maintain topological network continuity
        if best_voxel is None and len(causal_field.voxels) > 1:
            # Fallback to any existing voxel if no match score was above 0
            existing_ids = [vid for vid in causal_field.voxels if vid != new_id]
            if existing_ids:
                best_voxel = causal_field.voxels[existing_ids[0]]

        if best_voxel is not None and best_voxel.id in causal_field.voxels:
            causal_field.link_voxels(best_voxel.id, new_id, strength=harmony_score * 2.0)

        return new_voxel

    def execute_entropy_defense(self, thermal_stress: float, sensory_influx_rate: float):
        """
        [Entropy Defense Mechanism (엔트로피 방어 및 플라이휠 평형 유지)]

        When sensory influx explodes (high sensory_influx_rate & thermal_stress),
        prevents system deadlock by dynamically expanding the Local Zero domain radius
        and recalibrating resistance, ensuring the flywheel inertial momentum L stays balanced.
        """
        if thermal_stress > 2.0 or sensory_influx_rate > 5.0:
            # Expand Local Zero domain to absorb high energy influx without deadlock
            expansion_factor = 1.0 + 0.1 * min(10.0, thermal_stress + sensory_influx_rate)
            self.domain_radius *= expansion_factor

            # Damping adjustment to maintain flywheel rotational balance
            target_velocity = self.phase_velocity / expansion_factor
            self.phase_velocity = 0.9 * self.phase_velocity + 0.1 * target_velocity

    def get_state(self) -> Dict[str, Any]:
        """Returns diagnostic state dictionary."""
        return {
            "core_center": self.core_center.tolist(),
            "domain_radius": self.domain_radius,
            "phase_angle_rad": self.phase_angle,
            "phase_velocity_rad_s": self.phase_velocity,
            "current_resistance": self.current_resistance,
            "inward_tension_mag": self.inward_tension_mag,
            "flywheel_momentum": self.flywheel_inertia * self.phase_velocity,
            "resonant_harmony": self.resonant_harmony,
            "energy_efficiency": max(0.0, 1.0 - (self.current_resistance / self.base_resistance))
        }
