"""
Ecological Receptor Matrix (생태적 감각 수용체 매트릭스)

Converts multi-modal continuous environmental energy signals:
1. Audio Wave Spectrum (Frequency w & Amplitude A)
2. Optical Flow & Spatial Deformation (Spatial Wavevector k & Density Curvature)
3. Interaction Impedance & Latency Friction (Phase Shift phi & Damping Coefficient gamma)

into standardized physical continuous phase-wave field parameters.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Tuple
import math
import torch
import numpy as np


class SensoryModality(Enum):
    AUDIO = auto()
    VISUAL_OPTIC_FLOW = auto()
    IMPEDANCE_FRICTION = auto()


@dataclass
class ReceptorSignal:
    modality: SensoryModality
    amplitude: np.ndarray        # A_total
    frequency: float             # omega (rad/s or Hz normalized)
    wavevector: np.ndarray       # k (spatial direction & density)
    phase_shift: float           # phi (friction / impedance shift)
    damping: float               # gamma (friction attenuation coefficient)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContinuousWaveformTensor:
    field_dimension: int
    amplitude_tensor: torch.Tensor   # [dimension]
    frequency_tensor: torch.Tensor   # [dimension]
    wavevector_tensor: torch.Tensor  # [dimension, spatial_dim]
    phase_tensor: torch.Tensor       # [dimension]
    damping_tensor: torch.Tensor     # [dimension]


class EcologicalReceptorMatrix:
    """
    Standardizes environmental energy flow and system friction into continuous phase wave parameters.
    """

    def __init__(self, dimension: int = 64, spatial_dim: int = 3, dtype=torch.float32):
        self.dimension = dimension
        self.spatial_dim = spatial_dim
        self.dtype = dtype

        # Base characteristic carrier frequency spectrum per dimension
        # Allocating distinct characteristic frequencies omega_1, omega_2, ..., omega_N
        base_freqs = np.linspace(1.0, 100.0, num=self.dimension)
        self.base_frequency = torch.tensor(base_freqs, dtype=self.dtype)

        # Base wavevectors k_1, k_2, ..., k_N
        np.random.seed(42)
        base_k = np.random.randn(self.dimension, self.spatial_dim)
        base_k /= (np.linalg.norm(base_k, axis=1, keepdims=True) + 1e-8)
        self.base_wavevector = torch.tensor(base_k, dtype=self.dtype)

    def extract_audio_wave(self, audio_input: Any) -> ReceptorSignal:
        """
        Extracts frequency spectrum, amplitude, and acoustic energy from audio waveform/spectrum.
        """
        if isinstance(audio_input, dict):
            raw_spectrum = audio_input.get("spectrum", np.ones(self.dimension))
            amplitude_val = float(audio_input.get("amplitude", 1.0))
            fundamental_freq = float(audio_input.get("frequency", 440.0))
        elif isinstance(audio_input, (list, np.ndarray, torch.Tensor)):
            arr = np.asarray(audio_input, dtype=np.float32)
            if arr.ndim == 1:
                amplitude_val = float(np.mean(np.abs(arr))) + 1e-5
                fundamental_freq = float(np.argmax(np.abs(np.fft.rfft(arr))) + 1.0) * 10.0
                raw_spectrum = arr
            else:
                amplitude_val = float(np.linalg.norm(arr))
                fundamental_freq = 440.0
                raw_spectrum = arr.flatten()
        else:
            amplitude_val = float(audio_input)
            fundamental_freq = 440.0
            raw_spectrum = np.ones(self.dimension)

        if len(raw_spectrum) != self.dimension:
            raw_spectrum = np.interp(
                np.linspace(0, 1, self.dimension),
                np.linspace(0, 1, len(raw_spectrum)),
                raw_spectrum
            )

        amplitude = np.abs(raw_spectrum) * amplitude_val
        wavevector = np.zeros((self.dimension, self.spatial_dim), dtype=np.float32)
        wavevector[:, 0] = fundamental_freq / 100.0

        return ReceptorSignal(
            modality=SensoryModality.AUDIO,
            amplitude=amplitude,
            frequency=fundamental_freq,
            wavevector=wavevector,
            phase_shift=0.0,
            damping=0.01,
            metadata={"source": "audio_receptor"}
        )

    def extract_optic_flow(self, visual_input: Any) -> ReceptorSignal:
        """
        Extracts visual spatial flow, velocity vector, and spatial curvature density.
        """
        if isinstance(visual_input, dict):
            flow_vec = np.asarray(visual_input.get("optic_flow", [1.0, 0.0, 0.0]), dtype=np.float32)
            density = float(visual_input.get("density", 1.0))
            curvature = float(visual_input.get("curvature", 0.5))
        elif isinstance(visual_input, (list, np.ndarray, torch.Tensor)):
            arr = np.asarray(visual_input, dtype=np.float32).flatten()
            if len(arr) >= self.spatial_dim:
                flow_vec = arr[:self.spatial_dim]
            else:
                flow_vec = np.pad(arr, (0, self.spatial_dim - len(arr)))
            density = float(np.linalg.norm(flow_vec))
            curvature = 0.5
        else:
            flow_vec = np.ones(self.spatial_dim, dtype=np.float32)
            density = float(visual_input)
            curvature = 0.5

        if flow_vec.shape[0] != self.spatial_dim:
            flow_vec = np.resize(flow_vec, (self.spatial_dim,))

        # Spatial wavevector k derived from optic flow direction and spatial density
        k_direction = flow_vec / (np.linalg.norm(flow_vec) + 1e-8)
        wavevector = np.tile(k_direction * (1.0 + curvature), (self.dimension, 1))

        amplitude = np.full(self.dimension, density, dtype=np.float32)
        freq = float(np.linalg.norm(flow_vec) * 10.0 + 1.0)

        return ReceptorSignal(
            modality=SensoryModality.VISUAL_OPTIC_FLOW,
            amplitude=amplitude,
            frequency=freq,
            wavevector=wavevector,
            phase_shift=curvature * math.pi,
            damping=0.02,
            metadata={"source": "visual_receptor"}
        )

    def extract_impedance_friction(self, system_input: Any) -> ReceptorSignal:
        """
        Extracts system latency, thread contention, IO bottleneck as tactile/friction impedance.
        """
        if isinstance(system_input, dict):
            latency_ms = float(system_input.get("latency_ms", 10.0))
            thread_contention = float(system_input.get("contention", 0.1))
            io_impedance = float(system_input.get("impedance", 0.2))
        elif isinstance(system_input, (int, float)):
            latency_ms = float(system_input)
            thread_contention = 0.1
            io_impedance = latency_ms / 100.0
        else:
            latency_ms = 15.0
            thread_contention = 0.2
            io_impedance = 0.3

        phase_shift = (latency_ms / 1000.0) * (2.0 * math.pi)
        damping_gamma = io_impedance + thread_contention
        amplitude = np.full(self.dimension, thread_contention + 0.1, dtype=np.float32)

        wavevector = np.ones((self.dimension, self.spatial_dim), dtype=np.float32) * io_impedance

        return ReceptorSignal(
            modality=SensoryModality.IMPEDANCE_FRICTION,
            amplitude=amplitude,
            frequency=1.0 / (latency_ms / 1000.0 + 1e-4),
            wavevector=wavevector,
            phase_shift=phase_shift,
            damping=damping_gamma,
            metadata={"latency_ms": latency_ms, "source": "impedance_receptor"}
        )

    def process_and_pack(
        self,
        audio_input: Optional[Any] = None,
        visual_input: Optional[Any] = None,
        system_input: Optional[Any] = None
    ) -> ContinuousWaveformTensor:
        """
        Packs multi-sensory signals into a standardized continuous PyTorch wave parameter tensor.
        """
        signals: List[ReceptorSignal] = []
        if audio_input is not None:
            signals.append(self.extract_audio_wave(audio_input))
        if visual_input is not None:
            signals.append(self.extract_optic_flow(visual_input))
        if system_input is not None:
            signals.append(self.extract_impedance_friction(system_input))

        if not signals:
            # Fallback default idle noise signal
            signals.append(ReceptorSignal(
                modality=SensoryModality.IMPEDANCE_FRICTION,
                amplitude=np.ones(self.dimension, dtype=np.float32) * 0.1,
                frequency=1.0,
                wavevector=np.zeros((self.dimension, self.spatial_dim), dtype=np.float32),
                phase_shift=0.0,
                damping=0.01
            ))

        total_amp = torch.zeros(self.dimension, dtype=self.dtype)
        total_freq = self.base_frequency.clone()
        total_k = self.base_wavevector.clone()
        total_phase = torch.zeros(self.dimension, dtype=self.dtype)
        total_damping = torch.zeros(self.dimension, dtype=self.dtype)

        for sig in signals:
            amp_t = torch.tensor(sig.amplitude, dtype=self.dtype)
            total_amp += amp_t
            total_freq += torch.full_like(total_freq, sig.frequency * 0.05)
            total_k += torch.tensor(sig.wavevector, dtype=self.dtype) * 0.1
            total_phase += sig.phase_shift
            total_damping += sig.damping

        # Normalize wavevector
        k_norm = torch.norm(total_k, dim=-1, keepdim=True) + 1e-8
        total_k = total_k / k_norm

        return ContinuousWaveformTensor(
            field_dimension=self.dimension,
            amplitude_tensor=total_amp,
            frequency_tensor=total_freq,
            wavevector_tensor=total_k,
            phase_tensor=total_phase,
            damping_tensor=total_damping
        )
