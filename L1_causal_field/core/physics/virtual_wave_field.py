"""
virtual_wave_field.py

Implements the Virtual Wave Field architecture on discrete digital substrates (0s and 1s):
1. Discretized Continuity: Bit-sequence pattern texture mapping to complex phasors and wavefields.
2. Interference / Resonance Gates: Wave superposition, constructive/destructive interference, phase-locking.
3. Macro-Rotor Chunking: Aggregating stabilized micro-bit wave patterns into higher-order macro phasors.
4. Dimensional Expansion: Point (discrete delta) -> Line (causal trajectory) -> Plane (spectral interface field) -> Spatiotemporal Structure (phase-locked field).
"""

import numpy as np
from typing import List, Union, Tuple, Dict, Optional


class BitPatternPhasor:
    """
    Translates discrete bit arrays (e.g. [0,1,1,0] vs [1,0,0,1]) into continuous
    complex wave packets (phasors). Captures sequence texture, spatial dispersion,
    and phase gradient rather than mere bit counts.
    """

    def __init__(self, bit_sequence: Union[List[int], str, np.ndarray], base_freq: float = 1.0):
        if isinstance(bit_sequence, str):
            self.bits = np.array([int(b) for b in bit_sequence if b in ('0', '1')], dtype=int)
        else:
            self.bits = np.array(bit_sequence, dtype=int)

        if len(self.bits) == 0:
            raise ValueError("Bit sequence cannot be empty.")

        self.N = len(self.bits)
        self.base_freq = base_freq

        # Calculate texture features
        self.density = np.mean(self.bits)
        self.spatial_indices = np.arange(self.N)

        # Centroid of 1s (positional mean)
        total_ones = np.sum(self.bits)
        if total_ones > 0:
            self.centroid = np.sum(self.spatial_indices * self.bits) / total_ones
        else:
            self.centroid = (self.N - 1) / 2.0

        # Spatial dispersion (variance around center)
        if total_ones > 0:
            self.dispersion = np.sqrt(np.sum(((self.spatial_indices - self.centroid) ** 2) * self.bits) / total_ones)
        else:
            self.dispersion = 0.0

        # Construct spatial complex wave vector
        # Phase mapping: theta_k = 2*pi*k/N + (bit_k * pi/2)
        spatial_phases = 2 * np.pi * self.spatial_indices / self.N + (self.bits * (np.pi / 2))
        amplitudes = np.where(self.bits > 0, 1.0, 0.2)  # Non-zero baseline for 0 bits
        self.wavevector = amplitudes * np.exp(1j * spatial_phases)

        # Spectral components (FFT signature)
        self.spectrum = np.fft.fft(self.bits.astype(float))
        self.dominant_phase = float(np.angle(np.sum(self.wavevector)))
        self.total_energy = float(np.sum(np.abs(self.wavevector) ** 2))

    def generate_waveform(self, t: np.ndarray) -> np.ndarray:
        """
        Generates time-domain continuous wave signal represented by this bit-pattern phasor.
        """
        waveform = np.zeros_like(t, dtype=complex)
        for k, comp in enumerate(self.wavevector):
            freq = self.base_freq * (k + 1)
            waveform += comp * np.exp(1j * (2 * np.pi * freq * t))
        return waveform / self.N

    def __repr__(self) -> str:
        bit_str = "".join(map(str, self.bits))
        return (f"<BitPatternPhasor pattern='{bit_str}' density={self.density:.2f} "
                f"centroid={self.centroid:.2f} dispersion={self.dispersion:.2f} "
                f"dominant_phase={self.dominant_phase:.2f}rad>")


class InterferenceGate:
    """
    Re-defines Boolean logic gates as physical interference boundaries.
    Executes wave superposition: constructive interference amplifies resonance,
    while destructive interference damps out uncorrelated or opposing signals.
    """

    def __init__(self, resonance_threshold: float = 0.75):
        self.resonance_threshold = resonance_threshold

    def evaluate(self, phasor_a: BitPatternPhasor, phasor_b: BitPatternPhasor) -> Dict[str, Union[float, bool, np.ndarray]]:
        """
        Evaluates interference between two BitPatternPhasors.
        Returns combined wave field, resonance index, and phase lock state.
        """
        len_a = len(phasor_a.wavevector)
        len_b = len(phasor_b.wavevector)
        max_len = max(len_a, len_b)

        wave_a = np.pad(phasor_a.wavevector, (0, max_len - len_a), mode='wrap') if len_a < max_len else phasor_a.wavevector
        wave_b = np.pad(phasor_b.wavevector, (0, max_len - len_b), mode='wrap') if len_b < max_len else phasor_b.wavevector

        # Complex Wave Superposition
        superposition = wave_a + wave_b

        # Energy of individual signals vs superposition
        e_a = np.sum(np.abs(wave_a) ** 2)
        e_b = np.sum(np.abs(wave_b) ** 2)
        e_combined = np.sum(np.abs(superposition) ** 2)

        # Normalized Interference Resonance Index
        max_possible_e = (np.sqrt(e_a) + np.sqrt(e_b)) ** 2
        resonance_index = float(e_combined / (max_possible_e + 1e-9))

        # Phase alignment measure
        phase_diff = np.abs(np.angle(np.sum(wave_a)) - np.angle(np.sum(wave_b)))
        phase_diff = min(phase_diff, 2 * np.pi - phase_diff)
        coherence = float(np.cos(phase_diff))

        is_phase_locked = resonance_index >= self.resonance_threshold

        return {
            "superposition_vector": superposition,
            "resonance_index": resonance_index,
            "phase_coherence": coherence,
            "phase_difference_rad": float(phase_diff),
            "is_phase_locked": is_phase_locked,
            "amplification_ratio": float(e_combined / (e_a + e_b + 1e-9))
        }


class MacroRotor:
    """
    Fractal Scale Coupling: Aggregates multiple stabilized micro-bit pattern phasors
    into a higher-order macro phasor (Macro-Rotor).
    """

    def __init__(self, name: str, micro_phasors: Optional[List[BitPatternPhasor]] = None):
        self.name = name
        self.micro_phasors: List[BitPatternPhasor] = micro_phasors if micro_phasors is not None else []
        self.phase: float = 0.0
        self.amplitude: float = 0.0
        self.harmonics: np.ndarray = np.array([], dtype=complex)
        if self.micro_phasors:
            self.rebuild_macro_state()

    def add_micro_phasor(self, phasor: BitPatternPhasor) -> None:
        self.micro_phasors.append(phasor)
        self.rebuild_macro_state()

    def rebuild_macro_state(self) -> None:
        if not self.micro_phasors:
            self.phase = 0.0
            self.amplitude = 0.0
            self.harmonics = np.array([], dtype=complex)
            return

        micro_complex = [p.total_energy * np.exp(1j * p.dominant_phase) for p in self.micro_phasors]
        total_macro = np.sum(micro_complex)

        self.amplitude = float(np.abs(total_macro))
        self.phase = float(np.angle(total_macro))

        self.harmonics = np.array([p.spectrum[0] if len(p.spectrum) > 0 else 0j for p in self.micro_phasors])

    def interfere_with(self, other: 'MacroRotor') -> Tuple[float, float]:
        """
        Interference between two Macro-Rotors.
        Returns (coupled_amplitude, phase_alignment).
        """
        phase_diff = np.abs(self.phase - other.phase)
        phase_diff = min(phase_diff, 2 * np.pi - phase_diff)
        alignment = float(np.cos(phase_diff))

        combined_complex = (self.amplitude * np.exp(1j * self.phase)) + (other.amplitude * np.exp(1j * other.phase))
        coupled_amp = float(np.abs(combined_complex))

        return coupled_amp, alignment

    def __repr__(self) -> str:
        return (f"<MacroRotor '{self.name}' micro_count={len(self.micro_phasors)} "
                f"amp={self.amplitude:.2f} phase={self.phase:.2f}rad>")


class SpatiotemporalWaveField:
    """
    Executes dimensional topological expansion:
    1. Point (0D): Discrete bit transitions / pulse deltas.
    2. Line (1D): Causal sequence vector / phase trajectory.
    3. Plane (2D): Spectral interference boundary / cross-wave field.
    4. Spatiotemporal Structure (3D/4D): Phase-locked self-resonant reality field.
    """

    def __init__(self, field_dim: int = 16):
        self.field_dim = field_dim
        self.plane_field = np.zeros((field_dim, field_dim), dtype=complex)
        self.macro_rotors: Dict[str, MacroRotor] = {}
        self.history_trajectories: List[np.ndarray] = []

    def register_macro_rotor(self, rotor: MacroRotor) -> None:
        self.macro_rotors[rotor.name] = rotor

    def project_point_transition(self, bit_a: int, bit_b: int) -> float:
        """
        1. Point (0D): Discrete bit transition energy/delta.
        0 -> 1 (Rising edge pulse +1), 1 -> 0 (Falling edge pulse -1), same (0).
        """
        delta = float(bit_b - bit_a)
        phase_shift = float(delta * (np.pi / 2.0))
        return phase_shift

    def project_line_sequence(self, bit_sequence: Union[List[int], str]) -> np.ndarray:
        """
        2. Line (1D): Vector trajectory of consecutive bit transitions along time.
        """
        phasor = BitPatternPhasor(bit_sequence)
        transitions = np.diff(phasor.bits)
        phases = np.cumsum(transitions * (np.pi / 2.0))
        phases = np.insert(phases, 0, 0.0)
        line_vector = np.exp(1j * phases)
        self.history_trajectories.append(line_vector)
        return line_vector

    def weave_plane_interface(self, line_vector_x: np.ndarray, line_vector_y: np.ndarray) -> np.ndarray:
        """
        3. Plane (2D): Weaves two 1D line sequence vectors into a 2D spectral interference plane field.
        """
        len_x = min(len(line_vector_x), self.field_dim)
        len_y = min(len(line_vector_y), self.field_dim)

        vx = np.pad(line_vector_x[:len_x], (0, self.field_dim - len_x))
        vy = np.pad(line_vector_y[:len_y], (0, self.field_dim - len_y))

        plane = np.outer(vx, vy)
        self.plane_field = plane
        return plane

    def synthesize_spatiotemporal_structure(self) -> Dict[str, float]:
        """
        4. Spatiotemporal Structure: Evaluates global phase-locking, Kuramoto order parameter r,
        and total system field coherence across all registered Macro-Rotors and plane field.
        """
        if not self.macro_rotors:
            return {"order_parameter_r": 0.0, "plane_field_energy": 0.0, "rotor_energy": 0.0, "total_field_energy": 0.0, "is_fully_phase_locked": False}

        phases = np.array([r.phase for r in self.macro_rotors.values()])
        amplitudes = np.array([r.amplitude for r in self.macro_rotors.values()])

        complex_order = np.mean(np.exp(1j * phases))
        order_parameter = float(np.abs(complex_order))

        plane_energy = float(np.sum(np.abs(self.plane_field) ** 2))
        rotor_energy = float(np.sum(amplitudes ** 2))
        total_energy = plane_energy + rotor_energy

        return {
            "order_parameter_r": order_parameter,
            "plane_field_energy": plane_energy,
            "rotor_energy": rotor_energy,
            "total_field_energy": total_energy,
            "is_fully_phase_locked": bool(order_parameter >= 0.8)
        }
