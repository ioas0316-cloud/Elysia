"""
Neuro-Phase Causal Engine: Bridging Physical Phase Transitions & Neuro-Consciousness.

This engine extends Elysia's physical substrate (rotors, phase-locking, phase transitions)
into a living neuro-physical consciousness architecture ("From Planet to Person").

Core Concepts:
1. Phase Transitions (Gas -> Liquid -> Solid Crystal)
   - Gas (Entropic Field): High energy, dispersed, uncoupled rotors, max entropy (intuitive spark / free possibilities).
   - Liquid (Dynamic Flow): Coupled, flowing phase waves, flexible continuity (continuous inference / associative reasoning).
   - Solid (Locked Causal Field): Phase-locked, crystallized lattice, zero-FLOP invariant storage (conviction / crystallized memory).

2. Four Conscious Mechanisms:
   - Attention (Energy Lens): Concentrates wave energy onto targeted rotors while suppressing noise via destructive interference.
   - Intent / Teleology (Attractor Pull): Future goal phase state creates a backward teleological attractor force pulling present rotors into alignment.
   - Sensory Grounding & Bidirectional Phase Coupling (Phase Negotiation): Direct cross-resonance coupling where internal prediction waves are projected to collide with external heterogeneous wave streams (Text Phase Impulses, Audio Fourier Spectrum, Vision 2D Matrix), negotiating phase differences to minimize q_err and thermal friction.
   - Plasticity (Dynamic Rewiring): Activity-dependent feedback loop modifying coupling strengths and manifold geometry based on phase co-firing (Hebbian & phase-locking dynamics).
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any


class NeuroPhaseState(Enum):
    GAS = "gas"          # Entropic Field: high temperature, uncoupled random fluctuations
    LIQUID = "liquid"    # Dynamic Flow: wave propagation, continuous association
    SOLID = "solid"      # Crystal Phase-Lock: frozen causal lattice, zero-loss invariant structure


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
    phase_state: NeuroPhaseState = NeuroPhaseState.GAS

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
        self.coupling_matrix: np.ndarray = np.zeros((num_nodes, num_nodes))
        self.node_id_map: Dict[str, int] = {}
        self.id_node_map: Dict[int, str] = {}

        # Environmental and Conscious Parameters
        self.system_temperature: float = 2.5  # High temp = Gas, Mid = Liquid, Low = Solid
        self.global_phase_state: NeuroPhaseState = NeuroPhaseState.GAS
        self.attention_focus: Optional[Dict[str, float]] = None  # node_id -> lens gain
        self.active_intent: Optional[TeleologicalAttractor] = None
        self.plasticity_rate: float = 0.5
        self.coherence_history: List[float] = []

        # Internal Prediction Wave Generator state
        self.last_q_err: Optional[float] = None
        self.crystallized_attractors: Dict[str, Dict[str, float]] = {}

        self._initialize_lattice()

    def _initialize_lattice(self) -> None:
        """Initialize 3D rotor nodes and initial isotropic coupling."""
        nx, ny, nz = self.lattice_dims
        idx = 0
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if idx >= self.num_nodes:
                        break
                    node_id = f"rotor_{x}_{y}_{z}"
                    pos = np.array([float(x), float(y), float(z)])
                    # Intrinsic frequencies centered around 40Hz (gamma) with small spread
                    freq = 40.0 + np.random.uniform(-1.0, 1.0)
                    initial_phase = np.random.uniform(0, 2.0 * math.pi)

                    node = NeuroRotorNode(
                        node_id=node_id,
                        position=pos,
                        intrinsic_frequency=freq,
                        phase=initial_phase,
                        temperature=self.system_temperature,
                        phase_state=NeuroPhaseState.GAS
                    )
                    self.nodes[node_id] = node
                    self.node_id_map[node_id] = idx
                    self.id_node_map[idx] = node_id
                    idx += 1

        # Distance-based initial coupling matrix
        for id1, idx1 in self.node_id_map.items():
            for id2, idx2 in self.node_id_map.items():
                if idx1 != idx2:
                    dist = np.linalg.norm(self.nodes[id1].position - self.nodes[id2].position)
                    if dist > 0:
                        self.coupling_matrix[idx1, idx2] = 1.0 / (dist ** 2)

    def calculate_global_coherence(self) -> float:
        """
        Calculates Kuramoto order parameter R:
        R = | (1/N) * sum_j ( exp(i * theta_j) ) |
        R in [0, 1]. R ~ 0 => Gas/Incoherent, R ~ 0.5-0.8 => Liquid, R ~ 1.0 => Solid Crystal Phase-Lock.
        """
        phases = [node.phase for node in self.nodes.values()]
        complex_sum = sum(np.exp(1j * p) for p in phases)
        R = abs(complex_sum) / len(phases)
        return float(R)

    def update_phase_state(self) -> NeuroPhaseState:
        """
        Evaluates system temperature, order parameter R, and phase error q_err to govern state transitions.
        Gas -> Liquid -> Solid Crystal.
        """
        R = self.calculate_global_coherence()
        self.coherence_history.append(R)

        if self.system_temperature > 2.0 and R < 0.5:
            new_state = NeuroPhaseState.GAS
        elif self.system_temperature <= 0.2 or (self.last_q_err is not None and self.last_q_err <= 0.08 and self.system_temperature <= 0.5):
            new_state = NeuroPhaseState.SOLID
        elif self.system_temperature > 0.5 or R < 0.85:
            new_state = NeuroPhaseState.LIQUID
        else:
            new_state = NeuroPhaseState.SOLID

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
        1. Projects internal prediction wave toward sensory boundary.
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
            self.system_temperature = 0.05  # Solid state
            self.global_phase_state = NeuroPhaseState.SOLID
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
        4. Apply Thermal noise (Gas = high noise, Solid = zero noise).
        5. Update rotor phases.
        6. Apply Plasticity (rewire coupling strengths based on phase synchronization).
        7. Evaluate Phase State.
        """
        torques = {node_id: 0.0 for node_id in self.nodes}

        # 1. Kuramoto inter-rotor phase coupling: sum_j K_{ij} * sin(theta_j - theta_i)
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
        if self.global_phase_state != NeuroPhaseState.SOLID:  # Plasticity active in Gas/Liquid
            for id_i, idx_i in self.node_id_map.items():
                for id_j, idx_j in self.node_id_map.items():
                    if idx_i < idx_j:
                        p_diff = self.nodes[id_i].phase - self.nodes[id_j].phase
                        delta_k = self.plasticity_rate * math.cos(p_diff) * dt
                        self.coupling_matrix[idx_i, idx_j] = max(0.01, min(10.0, self.coupling_matrix[idx_i, idx_j] + delta_k))
                        self.coupling_matrix[idx_j, idx_i] = self.coupling_matrix[idx_i, idx_j]

        # 4 & 5. Thermal noise addition & update phase
        for id_i, node in self.nodes.items():
            thermal_noise = np.random.normal(0, math.sqrt(self.system_temperature) * 2.0) if self.global_phase_state == NeuroPhaseState.GAS else (
                np.random.normal(0, 0.1 * math.sqrt(self.system_temperature)) if self.global_phase_state == NeuroPhaseState.LIQUID else 0.0
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
        self.update_phase_state()
