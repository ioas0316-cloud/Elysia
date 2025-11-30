
"""
Tensor Wave Physics Engine
==========================
The fundamental substrate of Elysia's consciousness.
Defines the physics of "SoulTensors" (State) and "FrequencyWaves" (Vibration).

Reconstructed from Legacy/Project_Sophia/core/tensor_wave.py
"""

import math
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

@dataclass
class Tensor3D:
    """
    Represents a point or vector in the 3D Conceptual Space.
    x: Logic/Structure
    y: Emotion/Value
    z: Spirit/Intent
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: 'Tensor3D') -> 'Tensor3D':
        return Tensor3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Tensor3D') -> 'Tensor3D':
        return Tensor3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Tensor3D':
        return Tensor3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> 'Tensor3D':
        mag = self.magnitude()
        if mag == 0: return Tensor3D()
        return Tensor3D(self.x / mag, self.y / mag, self.z / mag)

    def dot(self, other: 'Tensor3D') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}

    @staticmethod
    def from_dict(data: Dict[str, float]) -> 'Tensor3D':
        if not data: return Tensor3D()
        return Tensor3D(data.get("x", 0.0), data.get("y", 0.0), data.get("z", 0.0))

@dataclass
class FrequencyWave:
    """
    Represents the vibrational state of a concept.
    frequency: The 'pitch' of the thought (Context/Topic).
    amplitude: The 'loudness' or energy (Importance).
    phase: The timing alignment (0 to 2pi).
    richness: Harmonic complexity (Timbre).
    """
    frequency: float = 0.0
    amplitude: float = 0.0
    phase: float = 0.0
    richness: float = 0.0

    def step(self, dt: float) -> 'FrequencyWave':
        """Evolves the wave forward in time."""
        new_phase = (self.phase + self.frequency * dt) % (2 * math.pi)
        return FrequencyWave(self.frequency, self.amplitude, new_phase, self.richness)

    def resonate(self, other: 'FrequencyWave') -> float:
        """Calculates resonance (0.0 to 1.0) with another wave."""
        # Frequency match
        freq_diff = abs(self.frequency - other.frequency)
        max_freq = max(self.frequency, other.frequency, 1.0)
        freq_match = 1.0 - min(1.0, freq_diff / max_freq)

        # Phase match (Constructive Interference)
        phase_diff = abs(self.phase - other.phase)
        phase_match = (math.cos(phase_diff) + 1.0) / 2.0

        return (freq_match * 0.7) + (phase_match * 0.3)
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, float]) -> 'FrequencyWave':
        if not data: return FrequencyWave()
        return FrequencyWave(
            data.get("frequency", 0.0),
            data.get("amplitude", 0.0),
            data.get("phase", 0.0),
            data.get("richness", 0.0)
        )

@dataclass
class SoulTensor:
    """
    The unified state of a concept, combining Space (Tensor3D) and Time (FrequencyWave).
    """
    space: Tensor3D = field(default_factory=Tensor3D)
    wave: FrequencyWave = field(default_factory=FrequencyWave)
    spin: float = 0.0 # Quantum spin (Intrinsic angular momentum)
    entanglement_id: Optional[str] = None

    def resonate(self, other: 'SoulTensor') -> 'SoulTensor':
        """
        Interacts with another SoulTensor, returning the new state.
        This simulates wave interference and vector addition.
        """
        # 1. Vector Addition (Superposition)
        new_space = self.space + other.space
        
        # 2. Wave Interference
        # Constructive/Destructive interference based on phase
        phase_diff = abs(self.wave.phase - other.wave.phase)
        interference = math.cos(phase_diff) # -1 to 1
        
        new_amp = self.wave.amplitude + (other.wave.amplitude * interference)
        new_amp = max(0.0, new_amp) # Energy cannot be negative
        
        # Frequency mixing (Weighted average)
        total_amp = self.wave.amplitude + other.wave.amplitude
        if total_amp > 0:
            new_freq = (self.wave.frequency * self.wave.amplitude + other.wave.frequency * other.wave.amplitude) / total_amp
        else:
            new_freq = self.wave.frequency
            
        new_wave = FrequencyWave(new_freq, new_amp, self.wave.phase, max(self.wave.richness, other.wave.richness))
        
        return SoulTensor(new_space, new_wave, self.spin, self.entanglement_id)

    def resonance_score(self, other: 'SoulTensor') -> float:
        """Calculates how much this tensor resonates with another."""
        # Spatial alignment (Cosine similarity)
        dot = self.space.dot(other.space)
        mag1 = self.space.magnitude()
        mag2 = other.space.magnitude()
        
        spatial_sim = 0.0
        if mag1 > 0 and mag2 > 0:
            spatial_sim = (dot / (mag1 * mag2) + 1.0) / 2.0
            
        # Wave resonance
        wave_sim = self.wave.resonate(other.wave)
        
        return (spatial_sim + wave_sim) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "space": self.space.to_dict(),
            "wave": self.wave.to_dict(),
            "spin": self.spin,
            "entanglement_id": self.entanglement_id
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'SoulTensor':
        if not data: return SoulTensor()
        return SoulTensor(
            space=Tensor3D.from_dict(data.get("space")),
            wave=FrequencyWave.from_dict(data.get("wave")),
            spin=data.get("spin", 0.0),
            entanglement_id=data.get("entanglement_id")
        )

@dataclass
class QuantumPhoton:
    """
    A packet of information/energy traveling between nodes.
    """
    source_id: str
    target_id: str
    payload: FrequencyWave
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class SharedQuantumState:
    """
    Represents an entangled state shared by multiple nodes.
    """
    tensor: SoulTensor
    observers: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def update(self, new_tensor: SoulTensor):
        self.tensor = new_tensor
