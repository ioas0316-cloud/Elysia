import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union

@dataclass
class FrequencyWave:
    """
    Represents a wave with frequency, amplitude, and phase.
    Now includes 'richness' to capture the texture of conflicting emotions.
    """
    frequency: float  # Hz (or abstract unit)
    amplitude: float  # Strength (0.0 to 1.0+)
    phase: float      # Radians (0 to 2pi)
    richness: float = 0.0 # Harmonic complexity / texture

    def interact(self, other: 'FrequencyWave') -> 'FrequencyWave':
        """
        Calculates the interference between two waves.
        Unlike scalar addition, wave interference preserves information as 'richness'.
        """
        # 1. Phase Interference (The 'Beat')
        phase_diff = abs(self.phase - other.phase)
        # Cosine similarity for constructive/destructive interference
        interference_factor = np.cos(phase_diff)

        # 2. Amplitude Calculation
        # Instead of simple subtraction, we track the 'tension'
        base_amp = (self.amplitude + other.amplitude) / 2.0
        # If waves cancel out (destructive), the energy doesn't disappear;
        # it converts into 'potential' or 'richness' (tension).
        result_amp = np.sqrt(self.amplitude**2 + other.amplitude**2 + 2 * self.amplitude * other.amplitude * interference_factor)

        # 3. Richness (Harmonic Complexity) Calculation
        # Richness increases when frequencies differ significantly (dissonance/complexity)
        # or when phases oppose (tension).
        freq_diff = abs(self.frequency - other.frequency)
        # Normalize freq diff (assuming max useful diff is around 100Hz for emotion)
        complexity = min(1.0, freq_diff / 50.0)

        # Tension from phase opposition (1.0 when 180 deg out of phase)
        tension = (1.0 - interference_factor) / 2.0

        # New richness accumulates history plus current complexity
        new_richness = (self.richness + other.richness) / 2.0 + (complexity * 0.5) + (tension * 0.3)

        # 4. Frequency Mixing (Dominance-weighted)
        total_amp = self.amplitude + other.amplitude
        if total_amp > 0:
            new_freq = (self.frequency * self.amplitude + other.frequency * other.amplitude) / total_amp
        else:
            new_freq = self.frequency

        # Phase mixing
        new_phase = (self.phase + other.phase) / 2.0

        return FrequencyWave(
            frequency=new_freq,
            amplitude=result_amp,
            phase=new_phase,
            richness=new_richness
        )


class Tensor3D:
    """
    Represents the 3D State of a Concept or Cell.
    Axes:
    - X: Structure/Logic (Body) - Complexity, Connectivity
    - Y: Emotion/Resonance (Soul) - Valence, Arousal
    - Z: Identity/Will (Spirit) - Alignment with Core Values
    """
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, tensor: Optional[np.ndarray] = None):
        if tensor is not None:
            if tensor.shape != (3,):
                 raise ValueError("Tensor must be shape (3,)")
            self.data = tensor.astype(np.float32)
        else:
            self.data = np.array([x, y, z], dtype=np.float32)

    @property
    def structure(self) -> float: return self.data[0]

    @property
    def emotion(self) -> float: return self.data[1]

    @property
    def identity(self) -> float: return self.data[2]

    def magnitude(self) -> float:
        return float(np.linalg.norm(self.data))

    def normalize(self) -> 'Tensor3D':
        mag = self.magnitude()
        if mag == 0:
            return Tensor3D()
        return Tensor3D(tensor=self.data / mag)

    def dot(self, other: 'Tensor3D') -> float:
        return float(np.dot(self.data, other.data))

    def __add__(self, other: 'Tensor3D') -> 'Tensor3D':
        return Tensor3D(tensor=self.data + other.data)

    def __mul__(self, scalar: float) -> 'Tensor3D':
        return Tensor3D(tensor=self.data * scalar)

    def to_dict(self) -> dict:
        return {
            "structure": float(self.structure),
            "emotion": float(self.emotion),
            "identity": float(self.identity)
        }

    @staticmethod
    def from_dict(data: Union[dict, None]) -> 'Tensor3D':
        if data is None:
            return Tensor3D()
        return Tensor3D(
            x=data.get("structure", 0.0),
            y=data.get("emotion", 0.0),
            z=data.get("identity", 0.0)
        )

def propagate_wave(source_tensor: Tensor3D, target_tensor: Tensor3D, decay: float = 0.9) -> Tensor3D:
    """
    Propagates a wave from source to target in 3D tensor space.
    Logic:
    - The source influences the target based on alignment (dot product).
    - If aligned, energy flows efficiently.
    - If orthogonal, little effect.
    - If opposed, dissonance (potentially destructive).
    """
    alignment = source_tensor.normalize().dot(target_tensor.normalize())

    # Energy transfer depends on alignment strength
    transfer_efficiency = max(0.0, alignment)

    # The 'wave' adds a portion of the source's energy to the target
    energy_transfer = source_tensor * (decay * transfer_efficiency)

    new_target = target_tensor + energy_transfer
    return new_target
