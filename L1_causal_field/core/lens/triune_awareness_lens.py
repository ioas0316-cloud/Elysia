"""
Triune Awareness Lens (삼위일체적 인지 렌즈)

Implements the fundamental 3D awareness axis (Cycle, Amplitude, Phase)
and continuous differential awareness (Sameness/Resonance vs Difference/Interference)
to serve as an observational cognitive axis rather than a static computational tool.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional


@dataclass
class TriuneSpectrum:
    """
    Represents the 3D Awareness Spectrum state perceived by the Triune Awareness Lens.
    """
    cycle: float        # Temporal loop / trajectory recurrence (Frequency/Period awareness)
    amplitude: float    # Energy / tension against internal resistance
    phase: float        # Relational angle / position in phase space [0, 2*pi)
    resonance: float    # Continuity / Sameness score (0.0 to 1.0)
    interference: float # Discontinuity / Difference score (0.0 to 1.0)
    delta_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": float(self.cycle),
            "amplitude": float(self.amplitude),
            "phase": float(self.phase),
            "resonance": float(self.resonance),
            "interference": float(self.interference),
            "delta_vector": self.delta_vector.tolist(),
            "metadata": self.metadata,
        }


class TriuneAwarenessLens:
    """
    Triune Awareness Lens (Awareness Unit / Seed Protocol Lens)

    Transforms raw experiential data streams into a 3D cognitive spectrum (Cycle, Amplitude, Phase)
    and discerns continuous sameness (resonance) versus difference (interference).
    """

    def __init__(
        self,
        base_resistance: float = 1.0,
        intrinsic_phase: float = 0.0,
        intrinsic_cycle: float = 1.0,
        sensitivity: float = 1.0,
    ):
        self.base_resistance = base_resistance
        self.intrinsic_phase = intrinsic_phase % (2 * np.pi)
        self.intrinsic_cycle = intrinsic_cycle
        self.sensitivity = sensitivity

        # Internal variable resistance tensor/matrix for dynamic calibration
        self.resistance_matrix = np.eye(3) * base_resistance
        self.phase_coupling_strength = 1.0

        # Historical trace mirror for cycle/recurrence tracking
        self.history_buffer = []
        self.max_history = 100

    def perceive(
        self,
        data_stream: np.ndarray,
        reference_waveform: Optional[np.ndarray] = None,
    ) -> TriuneSpectrum:
        """
        Processes an incoming data stream through the 3D Awareness Lens.

        Args:
            data_stream: 1D or 2D numpy array representing signal/sensory data.
            reference_waveform: Optional reference state array for differential comparison.

        Returns:
            TriuneSpectrum containing cycle, amplitude, phase, resonance, interference.
        """
        arr = np.atleast_1d(data_stream).astype(np.float64)
        if arr.size == 0:
            arr = np.zeros(1, dtype=np.float64)

        # 1. Amplitude (Energy / Tension awareness)
        # Tension is energy interacting with internal variable resistance
        raw_energy = np.linalg.norm(arr) / np.sqrt(arr.size)
        effective_resistance = float(np.trace(self.resistance_matrix) / 3.0)
        perceived_amplitude = raw_energy * effective_resistance * self.sensitivity

        # 2. Phase (Relational position / angle)
        # Compute dominant phase angle relative to intrinsic phase
        if arr.size >= 2:
            analytic_signal = arr[0] + 1j * arr[1]
            perceived_phase = np.angle(analytic_signal) % (2 * np.pi)
        else:
            perceived_phase = (arr[0] * np.pi) % (2 * np.pi)

        phase_delta = (perceived_phase - self.intrinsic_phase + np.pi) % (2 * np.pi) - np.pi

        # 3. Cycle (Temporal loop / recurrence awareness)
        self.history_buffer.append(arr.copy())
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)

        perceived_cycle = self._calculate_cycle_recurrence(arr)

        # 4. Continuous Differential Awareness (Sameness vs Difference / Resonance vs Interference)
        if reference_waveform is None:
            if len(self.history_buffer) >= 2:
                reference_waveform = self.history_buffer[-2]
            else:
                reference_waveform = arr

        resonance, interference = self._evaluate_differential_awareness(
            arr, reference_waveform, phase_delta
        )

        delta_vector = np.array([
            abs(perceived_cycle - self.intrinsic_cycle),
            perceived_amplitude,
            phase_delta,
        ])

        return TriuneSpectrum(
            cycle=float(perceived_cycle),
            amplitude=float(perceived_amplitude),
            phase=float(perceived_phase),
            resonance=float(resonance),
            interference=float(interference),
            delta_vector=delta_vector,
            metadata={
                "effective_resistance": effective_resistance,
                "intrinsic_phase": self.intrinsic_phase,
                "phase_coupling_strength": self.phase_coupling_strength,
            },
        )

    def _calculate_cycle_recurrence(self, current_arr: np.ndarray) -> float:
        """
        Detects recurrence trajectory (Period/Cycle) from history.
        """
        if len(self.history_buffer) < 3:
            return self.intrinsic_cycle

        # Measure autocorrelation / similarity with previous states in history
        similarities = []
        curr_norm = np.linalg.norm(current_arr) + 1e-9
        for prev_arr in reversed(self.history_buffer[:-1]):
            prev_norm = np.linalg.norm(prev_arr) + 1e-9
            # Pad or slice to match length
            min_len = min(current_arr.size, prev_arr.size)
            dot = np.dot(current_arr.flat[:min_len], prev_arr.flat[:min_len])
            sim = dot / (curr_norm * prev_norm)
            similarities.append(sim)

        # Period corresponds to lag with max similarity peak
        similarities = np.array(similarities)
        if len(similarities) > 0 and np.max(similarities) > 0.3:
            peak_lag = float(np.argmax(similarities) + 1)
            estimated_cycle = peak_lag / float(self.max_history) * 10.0
            return 0.7 * self.intrinsic_cycle + 0.3 * estimated_cycle

        return self.intrinsic_cycle

    def _evaluate_differential_awareness(
        self,
        arr: np.ndarray,
        ref_arr: np.ndarray,
        phase_delta: float,
    ) -> Tuple[float, float]:
        """
        Evaluates continuous resonance (sameness) and interference (difference).
        """
        min_len = min(arr.size, ref_arr.size)
        a = arr.flat[:min_len]
        b = ref_arr.flat[:min_len]

        norm_a = np.linalg.norm(a) + 1e-9
        norm_b = np.linalg.norm(b) + 1e-9

        cosine_sim = float(np.dot(a, b) / (norm_a * norm_b))
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

        # Phase alignment factor
        phase_alignment = np.cos(phase_delta)  # 1.0 when perfectly aligned, -1.0 when opposite

        # Continuous resonance is combination of directional alignment and phase coherence
        resonance = float(np.clip((cosine_sim + phase_alignment + 2.0) / 4.0, 0.0, 1.0))
        interference = float(1.0 - resonance)

        return resonance, interference
