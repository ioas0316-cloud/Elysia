"""
Neuro-Phase Causal Engine: Bridging Physical Phase Transitions & Neuro-Consciousness.

This engine extends Elysia's physical substrate (rotors, phase-locking, phase transitions)
into a living neuro-physical consciousness architecture ("From Planet to Person").

Core Concepts:
1. Phase Transitions (Gas -> Liquid -> Solid Crystal / ICE)
   - Gas (Entropic Field): High energy, dispersed, uncoupled rotors, max entropy (intuitive spark / free possibilities).
   - Liquid (Dynamic Flow): Coupled, flowing phase waves, flexible continuity (continuous inference / associative reasoning / Morphological Plasticity).
   - Solid / ICE (Locked Causal Field): Phase-locked, crystallized lattice, zero-FLOP invariant storage (conviction / crystallized memory / Invariant Anchor).

2. Dual-Axis Cognitive Causality:
   - Invariant Anchor (Fixed Axis / A_inv): Topological invariants (spectral graph Laplacian eigenvalues, 3D skeleton) preserved despite environmental strain.
   - Environmental Pressure Tensor (Variable Axis / P_env): Thermal noise, shear stress, directional flux, and boundary phase error (q_err).
   - Landau-Ginzburg Free Energy & Order Parameter (eta): Governs spontaneous phase transition dynamics.
   - Morphological Plasticity: Dynamic structural adaptation in liquid phase towards streamlined / functional shapes under environmental pressure.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any


class PhaseType(Enum):
    GAS = "gas"          # Entropic Field: high temperature / pressure, uncoupled random fluctuations
    LIQUID = "liquid"    # Dynamic Flow: wave propagation, continuous association, Morphological Plasticity
    ICE = "solid"        # Crystal Phase-Lock: frozen causal lattice, zero-loss invariant structure
    SOLID = "solid"      # Alias for ICE


# Backwards compatibility alias
NeuroPhaseState = PhaseType


@dataclass
class InvariantAnchor:
    """
    고정축 (Fixed Axis - A_inv):
    외계 변수 및 환경 압축 속에서도 보존되는 본질적 위상학적/구조적 불변 매니폴드.
    """
    num_nodes: int
    topo_matrix: np.ndarray          # Complex or float matrix (N, N): 본질적 복소 연결 가중치
    spectral_invariants: np.ndarray  # Float array (K,): 라플라시안 고유값 불변량
    geometric_skeleton: np.ndarray   # Float matrix (N, 3): 3D 불변 골격/앵커 좌표


@dataclass
class EnvironmentalPressure:
    """
    변수축 (Variable Axis - P_env):
    외부 환경이 내계 경계면에 가하는 물리적 압력 및 마찰 텐서.
    """
    thermal_noise: float = 1.0              # 열적 노이즈 (온도 T)
    shear_stress: float = 0.0               # 전단 마찰력 / 유체 저항 (tau)
    directional_flux: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )  # 외부 파동 유입 방향 벡터 (J)
    phase_error: float = 0.0                # 위상 오차 (q_err)

    @property
    def magnitude(self) -> float:
        """환경 압력의 총합 스칼라 강도 계산"""
        flux_mag = float(np.linalg.norm(self.directional_flux))
        return float(self.thermal_noise + self.shear_stress + flux_mag + (self.phase_error * 2.0))


@dataclass
class PhaseState:
    """
    시스템의 현재 상전이 상태, 질서 파라미터(eta), 및 자유 에너지 메트릭.
    """
    order_parameter: float            # eta in [0, 1]
    current_phase: PhaseType          # GAS, LIQUID, ICE
    free_energy: float                # Landau-Ginzburg 자유 에너지 F
    rotor_phases: np.ndarray          # Float array (N,): 현재 로터 개체들의 위상 theta_i


class NeuroPhaseCausalMap:
    """
    고정축(Invariant Anchor)과 변수축(Environmental Pressure) 간의
    상전이 궤적을 관측하고 수렴을 유도하는 위상 지도 엔진.
    """
    def __init__(
        self,
        anchor: InvariantAnchor,
        critical_pressure: float = 2.5,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5
    ):
        self.anchor = anchor
        self.p_crit = critical_pressure
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_free_energy(
        self,
        eta: float,
        pressure: EnvironmentalPressure,
        rotor_weights: np.ndarray
    ) -> float:
        """Landau-Ginzburg 자유 에너지 포텐셜 F(eta) 계산"""
        p_mag = pressure.magnitude
        # A_inv와 현재 로터 가중치 간의 내적 (구조적 일치도)
        alignment = float(np.abs(np.trace(np.dot(rotor_weights, np.conjugate(self.anchor.topo_matrix.T)))))

        f_val = (self.alpha * (self.p_crit - p_mag) * (eta ** 2)) + \
                (self.beta * (eta ** 4)) - \
                (self.gamma * alignment * eta)
        return float(f_val)

    def evaluate_phase_transition(
        self,
        rotor_phases: np.ndarray,
        rotor_weights: np.ndarray,
        pressure: EnvironmentalPressure
    ) -> PhaseState:
        """
        현재 로터 상태와 환경 압력을 수용하여
        상전이(GAS -> LIQUID -> ICE) 및 질서 파라미터를 판별하는 핵심 함수.
        """
        N = len(rotor_phases)
        if N == 0:
            return PhaseState(0.0, PhaseType.GAS, 0.0, rotor_phases)

        complex_rotors = np.exp(1j * rotor_phases)
        eta = float(np.abs(np.mean(complex_rotors)))

        f_val = self.compute_free_energy(eta, pressure, rotor_weights)
        p_mag = pressure.magnitude

        if p_mag > self.p_crit:
            # Extreme environmental pressure / thermal noise breaks phase lock -> GAS
            phase = PhaseType.GAS if eta < 0.8 else PhaseType.LIQUID
        elif p_mag <= 0.4 or eta >= 0.85:
            phase = PhaseType.ICE
        elif p_mag <= self.p_crit or (0.25 <= eta < 0.85):
            phase = PhaseType.LIQUID
        else:
            phase = PhaseType.GAS

        return PhaseState(
            order_parameter=eta,
            current_phase=phase,
            free_energy=f_val,
            rotor_phases=rotor_phases
        )

    def adapt_morphology(
        self,
        current_coords: np.ndarray,
        pressure: EnvironmentalPressure,
        phase_state: PhaseState
    ) -> np.ndarray:
        """
        형태적 가소성(Morphological Plasticity):
        환경 저항(변수축)에 맞서 불변 뼈대(고정축)를 유지하며 형태를 수렴/변형하는 연산.
        """
        if phase_state.current_phase == PhaseType.GAS:
            # 기체상: 고엔트로피 무작위 탐색 (확산)
            noise = np.random.randn(*current_coords.shape) * 0.1 * max(0.1, pressure.thermal_noise)
            return current_coords + noise

        elif phase_state.current_phase == PhaseType.LIQUID:
            # 액체상: 환경 유체 저항(Flux/Shear)에 반응하여 유선형/날개 궤적으로 형태 이행
            flux = pressure.directional_flux
            if flux.ndim == 1 and current_coords.ndim == 2:
                flux = np.tile(flux, (current_coords.shape[0], 1))
            target_skeletal_alignment = self.anchor.geometric_skeleton - current_coords

            # 유선형 수렴 벡터 = (환경 압력 저항 벡터) + (고정축 복원력)
            morph_drift = (0.6 * flux) + (0.4 * target_skeletal_alignment)
            return current_coords + (0.05 * morph_drift)

        else: # PhaseType.ICE / SOLID
            # 고체상: 고정축(Attractor Anchor)에 완전히 결빙 및 위상 고정 (변형 최소화)
            return 0.95 * current_coords + 0.05 * self.anchor.geometric_skeleton


@dataclass
class ExternalWaveStream:
    """
    Represents an external wave stream penetrating the sensory boundary.
    Supports 3 modalities treated as physical waves rather than discrete file packets:
    - Text: Phase impulse array (discrete phase shocks)
    - Audio: Frequency spectrum (Fourier pressure waves)
    - Vision: 2D electromagnetic phase matrix
    """
    modality: str  # 'text', 'audio', 'vision'
    wave_phases: np.ndarray  # Array of phase angles or 2D matrix
    frequencies: np.ndarray  # Corresponding frequencies
    amplitude: float = 1.0


@dataclass
class NeuroRotorNode:
    """
    Represents an atomic microscopic oscillator (neuronal/synaptic rotor) in 3D lattice space.
    Holds intrinsic frequency, phase angle, angular velocity, and activation energy.
    """
    node_id: str
    position: np.ndarray  # 3D coordinate (x, y, z)
    intrinsic_frequency: float  # Spontaneous natural frequency (e.g., 40Hz gamma, 10Hz alpha)
    phase: float = 0.0          # Current phase angle theta in [0, 2pi)
    angular_velocity: float = 0.0
    energy: float = 1.0         # Activation energy / amplitude
    temperature: float = 1.0    # Local thermal fluctuation level
    phase_state: PhaseType = PhaseType.GAS

    def update_phase(self, dt: float, external_torque: float = 0.0) -> None:
        """Advance phase based on intrinsic frequency, velocity, and external torque/coupling."""
        self.angular_velocity = 2.0 * math.pi * self.intrinsic_frequency + external_torque
        self.phase = (self.phase + self.angular_velocity * dt) % (2.0 * math.pi)


@dataclass
class TeleologicalAttractor:
    """
    Represents a future target goal state (Intent) acting as a teleological attractor force.
    """
    target_id: str
    target_phases: Dict[str, float]  # target phase angle per rotor node ID
    attractor_strength: float = 1.5


@dataclass
class PhaseNegotiationResult:
    """
    Result of bidirectional phase negotiation between internal prediction wave and external wave stream.
    """
    q_err: float                    # Total phase error scalar
    thermal_friction: float         # Thermal boundary friction
    resonance_level: float          # Coherence / Resonance [0, 1]
    is_crystallized: bool           # True if q_err < threshold (Phase-locked ICE)
    internal_prediction_wave: np.ndarray
    external_wave: np.ndarray


class NeuroPhaseCausalEngine:
    """
    The main engine executing phase transitions, bidirectional phase negotiation, and neuro-conscious dynamics.
    """

    def __init__(self, num_nodes: int = 16, lattice_dims: Tuple[int, int, int] = (4, 2, 2)):
        self.num_nodes = num_nodes
        self.lattice_dims = lattice_dims
        self.nodes: Dict[str, NeuroRotorNode] = {}
        self.coupling_matrix: np.ndarray = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        self.node_id_map: Dict[str, int] = {}
        self.id_node_map: Dict[int, str] = {}

        # Environmental and Conscious Parameters
        self.system_temperature: float = 2.5  # High temp = Gas, Mid = Liquid, Low = Solid
        self.global_phase_state: PhaseType = PhaseType.GAS
        self.attention_focus: Optional[Dict[str, float]] = None  # node_id -> lens gain
        self.active_intent: Optional[TeleologicalAttractor] = None
        self.plasticity_rate: float = 0.5
        self.coherence_history: List[float] = []

        # Internal Prediction Wave Generator state
        self.last_q_err: Optional[float] = None
        self.crystallized_attractors: Dict[str, Dict[str, float]] = {}

        # Dual-Axis Causal Framework
        self.pressure = EnvironmentalPressure(thermal_noise=self.system_temperature)
        self.anchor: Optional[InvariantAnchor] = None
        self.causal_map: Optional[NeuroPhaseCausalMap] = None

        self._initialize_lattice()

    def _initialize_lattice(self) -> None:
        """Initialize 3D rotor nodes, invariant anchor skeleton, and initial coupling matrix."""
        nx, ny, nz = self.lattice_dims
        idx = 0
        skeleton_coords = []
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if idx >= self.num_nodes:
                        break
                    node_id = f"rotor_{x}_{y}_{z}"
                    pos = np.array([float(x), float(y), float(z)], dtype=np.float64)
                    skeleton_coords.append(pos)
                    # Intrinsic frequencies centered around 40Hz (gamma) with small spread
                    freq = 40.0 + np.random.uniform(-1.0, 1.0)
                    initial_phase = np.random.uniform(0, 2.0 * math.pi)

                    node = NeuroRotorNode(
                        node_id=node_id,
                        position=pos,
                        intrinsic_frequency=freq,
                        phase=initial_phase,
                        temperature=self.system_temperature,
                        phase_state=PhaseType.GAS
                    )
                    self.nodes[node_id] = node
                    self.node_id_map[node_id] = idx
                    self.id_node_map[idx] = node_id
                    idx += 1

        skeleton_mat = np.array(skeleton_coords, dtype=np.float64)

        # Distance-based initial coupling matrix
        for id1, idx1 in self.node_id_map.items():
            for id2, idx2 in self.node_id_map.items():
                if idx1 != idx2:
                    dist = np.linalg.norm(self.nodes[id1].position - self.nodes[id2].position)
                    if dist > 0:
                        self.coupling_matrix[idx1, idx2] = 1.0 / (dist ** 2)

        # Compute Spectral Graph Laplacian for Invariant Anchor
        degree_mat = np.diag(np.sum(self.coupling_matrix, axis=1))
        laplacian = degree_mat - self.coupling_matrix
        eigenvalues = np.linalg.eigvalsh(laplacian)

        self.anchor = InvariantAnchor(
            num_nodes=self.num_nodes,
            topo_matrix=self.coupling_matrix.astype(complex),
            spectral_invariants=eigenvalues,
            geometric_skeleton=skeleton_mat
        )
        self.causal_map = NeuroPhaseCausalMap(anchor=self.anchor)

    def calculate_global_coherence(self) -> float:
        """
        Calculates Kuramoto order parameter eta (R):
        eta = | (1/N) * sum_j ( exp(i * theta_j) ) |
        eta in [0, 1]. eta ~ 0 => Gas/Incoherent, eta ~ 0.3-0.8 => Liquid, eta ~ 0.8-1.0 => ICE.
        """
        phases = [node.phase for node in self.nodes.values()]
        complex_sum = sum(np.exp(1j * p) for p in phases)
        eta = abs(complex_sum) / len(phases) if phases else 0.0
        return float(eta)

    def update_phase_state(self) -> PhaseType:
        """
        Evaluates environmental pressure tensor and order parameter eta to govern state transitions
        via Landau-Ginzburg free energy evaluation.
        Gas -> Liquid -> Solid Crystal (ICE).
        """
        self.pressure.thermal_noise = self.system_temperature
        if self.last_q_err is not None:
            self.pressure.phase_error = self.last_q_err

        rotor_phases = np.array([self.nodes[nid].phase for nid in sorted(self.nodes.keys())])

        if self.causal_map:
            p_state = self.causal_map.evaluate_phase_transition(
                rotor_phases=rotor_phases,
                rotor_weights=self.coupling_matrix,
                pressure=self.pressure
            )
            new_state = p_state.current_phase
            coherence = p_state.order_parameter
        else:
            coherence = self.calculate_global_coherence()
            if self.system_temperature > 2.0 and coherence < 0.5:
                new_state = PhaseType.GAS
            elif self.system_temperature <= 0.2 or (self.last_q_err is not None and self.last_q_err <= 0.08 and self.system_temperature <= 0.5):
                new_state = PhaseType.ICE
            elif self.system_temperature > 0.5 or coherence < 0.85:
                new_state = PhaseType.LIQUID
            else:
                new_state = PhaseType.ICE

        self.coherence_history.append(coherence)
        self.global_phase_state = new_state
        for node in self.nodes.values():
            node.phase_state = new_state
            node.temperature = self.system_temperature
        return new_state

    def apply_attention_lens(self, target_nodes: List[str], gain: float = 3.0) -> None:
        """
        Mechanism 1: Attention (Energy Lens).
        Focuses energy into specific target rotors, amplifying their signal while suppressing background noise.
        """
        self.attention_focus = {}
        for node_id in self.nodes:
            if node_id in target_nodes:
                self.attention_focus[node_id] = gain
                self.nodes[node_id].energy *= gain
            else:
                self.attention_focus[node_id] = 0.2  # Suppress surrounding background noise
                self.nodes[node_id].energy *= 0.8

    def set_teleological_intent(self, target_phases: Dict[str, float], strength: float = 5.0) -> None:
        """
        Mechanism 2: Intent / Teleology (Attractor Vector).
        Establishes a future target phase attractor that exerts backward pull on current rotor phases.
        """
        self.active_intent = TeleologicalAttractor(
            target_id="intent_goal",
            target_phases=target_phases,
            attractor_strength=strength
        )

    def project_internal_prediction_wave(self) -> np.ndarray:
        """
        Generates internal prediction wave projected outward from current rotor lattice state.
        theta_pred_i = phase_i
        """
        pred_phases = []
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            pred_phases.append(node.phase)
        return np.array(pred_phases, dtype=np.float64)

    def negotiate_bidirectional_phase(
        self,
        external_stream: ExternalWaveStream,
        coupling_gain: float = 1.5,
        crystallization_threshold: float = 0.08
    ) -> PhaseNegotiationResult:
        """
        Executes Bidirectional Phase Coupling & Negotiation:
        1. Projects internal prediction wave toward sensory boundary (Markov Blanket).
        2. Clashes internal wave against external wave stream.
        3. Computes phase error q_err = mean( |sin((theta_ext - theta_pred)/2)| ).
        4. Entrains internal rotors toward external stream while pushing back external stream (coupling).
        5. Cools system temperature as q_err drops (dissipates thermal friction).
        6. Reaches Solid Crystal (ICE) state when q_err < threshold.
        """
        pred_wave = self.project_internal_prediction_wave()
        ext_wave = external_stream.wave_phases.flatten()
        ext_freqs = external_stream.frequencies.flatten() if len(external_stream.frequencies) > 0 else np.full(len(pred_wave), 40.0)

        # Resample external wave & freqs to match internal node count if dimensions differ
        if len(ext_wave) != len(pred_wave):
            ext_wave = np.interp(
                np.linspace(0, len(ext_wave) - 1, len(pred_wave)),
                np.arange(len(ext_wave)),
                ext_wave
            )
            ext_freqs = np.interp(
                np.linspace(0, len(ext_freqs) - 1, len(pred_wave)),
                np.arange(len(ext_freqs)),
                ext_freqs
            )

        # 3. Compute Phase Error (q_err) and boundary friction
        phase_diffs = np.abs(np.sin((ext_wave - pred_wave) / 2.0))
        q_err = float(np.mean(phase_diffs))
        thermal_friction = q_err * self.system_temperature

        # Update variable axis pressure
        self.pressure.phase_error = q_err
        self.pressure.shear_stress = float(np.std(phase_diffs))

        # 4. Bidirectional Entrainment (Entraining internal rotors to external wave & frequency)
        sorted_keys = sorted(self.nodes.keys())
        for idx, node_id in enumerate(sorted_keys):
            node = self.nodes[node_id]
            delta_p = math.sin(ext_wave[idx] - node.phase)
            # Entrain phase and intrinsic frequency toward external wave
            node.phase = (node.phase + coupling_gain * delta_p * 0.4) % (2.0 * math.pi)
            node.intrinsic_frequency = (1.0 - 0.2 * coupling_gain) * node.intrinsic_frequency + (0.2 * coupling_gain) * ext_freqs[idx]
            node.energy += 0.2 * (1.0 - phase_diffs[idx])

        # 5. Dissipate system temperature in proportion to alignment (cooling)
        cooling_factor = 0.70 if q_err < 0.2 else 0.88
        self.system_temperature = max(0.05, self.system_temperature * cooling_factor)
        self.last_q_err = q_err

        # Check crystallization
        is_crystallized = (q_err < crystallization_threshold) or (self.system_temperature <= 0.1)
        if is_crystallized:
            self.system_temperature = 0.05  # Solid / ICE state
            self.global_phase_state = PhaseType.ICE
            # Store concept as phase-locked attractor basin
            concept_key = f"{external_stream.modality}_concept"
            self.crystallized_attractors[concept_key] = {
                nid: self.nodes[nid].phase for nid in self.nodes
            }

        self.update_phase_state()
        resonance = 1.0 - q_err

        return PhaseNegotiationResult(
            q_err=q_err,
            thermal_friction=thermal_friction,
            resonance_level=resonance,
            is_crystallized=is_crystallized,
            internal_prediction_wave=pred_wave,
            external_wave=ext_wave
        )

    def adapt_morphology(self) -> np.ndarray:
        """
        Executes Morphological Plasticity step based on current phase state and environmental pressure.
        Updates 3D positions of rotor nodes.
        """
        if not self.causal_map:
            return np.array([node.position for node in self.nodes.values()])

        current_coords = np.array([self.nodes[nid].position for nid in sorted(self.nodes.keys())])
        rotor_phases = np.array([self.nodes[nid].phase for nid in sorted(self.nodes.keys())])

        p_state = self.causal_map.evaluate_phase_transition(
            rotor_phases=rotor_phases,
            rotor_weights=self.coupling_matrix,
            pressure=self.pressure
        )

        new_coords = self.causal_map.adapt_morphology(
            current_coords=current_coords,
            pressure=self.pressure,
            phase_state=p_state
        )

        for idx, nid in enumerate(sorted(self.nodes.keys())):
            self.nodes[nid].position = new_coords[idx]

        return new_coords

    def inject_sensory_grounding(self, external_wave: Dict[str, float], coupling_gain: float = 2.0) -> None:
        """
        Mechanism 3: Sensory Grounding (Cross-Resonance).
        Injects external wave phases directly into matching internal rotor nodes, forcing cross-resonance.
        """
        for node_id, ext_phase in external_wave.items():
            if node_id in self.nodes:
                node = self.nodes[node_id]
                phase_diff = math.sin(ext_phase - node.phase)
                node.phase = (node.phase + coupling_gain * phase_diff * 0.2) % (2.0 * math.pi)
                node.energy += 0.5

    def step(self, dt: float = 0.001) -> Dict[str, float]:
        """
        Advances the simulation by 1 time step dt:
        1. Compute Kuramoto inter-rotor coupling torques.
        2. Apply Teleological Intent attractor force.
        3. Apply Attention gain / damping.
        4. Apply Thermal noise (Gas = high noise, ICE = zero noise).
        5. Update rotor phases.
        6. Apply Plasticity (rewire coupling strengths & morph coordinates).
        7. Evaluate Phase State.
        """
        torques = {node_id: 0.0 for node_id in self.nodes}

        # 1. Kuramoto inter-rotor phase coupling
        for id_i, idx_i in self.node_id_map.items():
            node_i = self.nodes[id_i]
            for id_j, idx_j in self.node_id_map.items():
                if idx_i != idx_j:
                    node_j = self.nodes[id_j]
                    K_ij = self.coupling_matrix[idx_i, idx_j]
                    phase_diff = node_j.phase - node_i.phase
                    torques[id_i] += K_ij * math.sin(phase_diff) * node_j.energy

        # 2. Teleological Intent Pull (Attractor Force)
        if self.active_intent:
            for id_i, target_phase in self.active_intent.target_phases.items():
                if id_i in self.nodes:
                    node_i = self.nodes[id_i]
                    phase_diff = math.sin(target_phase - node_i.phase)
                    pull_torque = self.active_intent.attractor_strength * phase_diff
                    torques[id_i] += pull_torque
                    node_i.phase = (node_i.phase + 0.2 * phase_diff) % (2.0 * math.pi)

        # 3. Attention Lens Damping/Amplification adjustment
        if self.attention_focus:
            for id_i, gain in self.attention_focus.items():
                torques[id_i] *= gain

        # 6. Plasticity: Hebbian phase-locking coupling adaptation before phase update
        if self.global_phase_state != PhaseType.ICE:  # Plasticity active in Gas/Liquid
            for id_i, idx_i in self.node_id_map.items():
                for id_j, idx_j in self.node_id_map.items():
                    if idx_i < idx_j:
                        p_diff = self.nodes[id_i].phase - self.nodes[id_j].phase
                        delta_k = self.plasticity_rate * math.cos(p_diff) * dt
                        self.coupling_matrix[idx_i, idx_j] = max(0.01, min(10.0, self.coupling_matrix[idx_i, idx_j] + delta_k))
                        self.coupling_matrix[idx_j, idx_i] = self.coupling_matrix[idx_i, idx_j]

            # Execute morphological coordinate adaptation
            self.adapt_morphology()

        # 4 & 5. Thermal noise addition & update phase
        for id_i, node in self.nodes.items():
            thermal_noise = np.random.normal(0, math.sqrt(self.system_temperature) * 2.0) if self.global_phase_state == PhaseType.GAS else (
                np.random.normal(0, 0.1 * math.sqrt(self.system_temperature)) if self.global_phase_state == PhaseType.LIQUID else 0.0
            )
            total_torque = torques[id_i] + thermal_noise
            node.update_phase(dt, external_torque=total_torque)

        # 7. Update Phase State
        current_state = self.update_phase_state()
        coherence = self.calculate_global_coherence()

        return {
            "coherence": coherence,
            "temperature": self.system_temperature,
            "q_err": self.last_q_err if self.last_q_err is not None else 0.0,
            "state": current_state.value
        }

    def set_temperature(self, temp: float) -> None:
        """Sets external temperature to induce cooling/heating phase transitions."""
        self.system_temperature = temp
        self.pressure.thermal_noise = temp
        self.update_phase_state()
