import numpy as np
import uuid
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Union, Dict

@dataclass
class FrequencyWave:
    """
    Represents the 'Wave' aspect of the SoulTensor.
    Controls Time, Motion, and Color (Frequency).
    """
    frequency: float  # Hz: Determines the 'Color' or 'Type' of emotion
    amplitude: float  # Magnitude: The intensity or energy
    phase: float      # Radians (0-2pi): The 'Timing' or 'Spin' state
    richness: float = 0.0 # Harmonic complexity (Texture)

    def step(self, dt: float = 0.1) -> 'FrequencyWave':
        """
        Advances the wave in time.
        The phase rotates based on frequency (Spin).
        """
        # d(phase)/dt = frequency * 2pi
        # New phase = current + angular_velocity * dt
        new_phase = (self.phase + self.frequency * 2 * np.pi * dt) % (2 * np.pi)
        return FrequencyWave(self.frequency, self.amplitude, new_phase, self.richness)

    def interact(self, other: 'FrequencyWave') -> 'FrequencyWave':
        """
        Calculates the interference between two waves.
        """
        # 1. Phase Interference (The 'Chemistry')
        phase_diff = abs(self.phase - other.phase)
        interference_factor = np.cos(phase_diff) # 1.0 (In-phase) to -1.0 (Out-phase)

        # 2. Amplitude Calculation (Energy summation with interference)
        # Resultant Amplitude A = sqrt(A1^2 + A2^2 + 2*A1*A2*cos(delta_phi))
        new_amp_sq = self.amplitude**2 + other.amplitude**2 + 2 * self.amplitude * other.amplitude * interference_factor
        result_amp = np.sqrt(max(0.0, new_amp_sq))

        # 3. Richness (Texture)
        # Dissonance adds richness.
        freq_diff = abs(self.frequency - other.frequency)
        dissonance = min(1.0, freq_diff / 100.0) # Assume 100Hz diff is max dissonance
        tension = (1.0 - interference_factor) / 2.0 # 0 to 1

        new_richness = (self.richness + other.richness) / 2.0 + (dissonance * 0.4) + (tension * 0.2)

        # 4. Frequency Mixing (Center of Gravity by Amplitude)
        total_amp = self.amplitude + other.amplitude
        if total_amp > 0:
            new_freq = (self.frequency * self.amplitude + other.frequency * other.amplitude) / total_amp
        else:
            new_freq = (self.frequency + other.frequency) / 2.0

        # Phase mixing (Weighted circular mean)
        # Simplified: average phase
        new_phase = (self.phase + other.phase) / 2.0

        return FrequencyWave(new_freq, result_amp, new_phase, new_richness)

    def to_dict(self):
        return {
            'frequency': self.frequency,
            'amplitude': self.amplitude,
            'phase': self.phase,
            'richness': self.richness
        }

    @staticmethod
    def from_dict(data):
        if not data: return FrequencyWave(0.0, 0.0, 0.0)
        return FrequencyWave(
            data.get('frequency', 0.0),
            data.get('amplitude', 0.0),
            data.get('phase', 0.0),
            data.get('richness', 0.0)
        )

@dataclass
class Tensor3D:
    """
    Represents the 'Space' aspect of the SoulTensor.
    Axes: Body (X), Soul (Y), Spirit (Z).
    """
    x: float = 0.0 # Structure / Mass
    y: float = 0.0 # Emotion / Energy
    z: float = 0.0 # Identity / Will

    @property
    def magnitude(self) -> float:
        return float(np.sqrt(self.x**2 + self.y**2 + self.z**2))

    def normalize(self) -> 'Tensor3D':
        mag = self.magnitude
        if mag == 0: return Tensor3D()
        return Tensor3D(self.x/mag, self.y/mag, self.z/mag)

    def dot(self, other: 'Tensor3D') -> float:
        return self.x*other.x + self.y*other.y + self.z*other.z

    def __add__(self, other: 'Tensor3D') -> 'Tensor3D':
        return Tensor3D(self.x+other.x, self.y+other.y, self.z+other.z)

    def __mul__(self, scalar: float) -> 'Tensor3D':
        return Tensor3D(self.x*scalar, self.y*scalar, self.z*scalar)

    def to_dict(self):
        return {'x': self.x, 'y': self.y, 'z': self.z}

    @staticmethod
    def from_dict(data):
        if not data: return Tensor3D()
        return Tensor3D(data.get('x', 0.0), data.get('y', 0.0), data.get('z', 0.0))

@dataclass
class QuantumPhoton:
    """
    A particle of information (Active Data).
    Carries momentum and a wave payload.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: FrequencyWave = field(default_factory=lambda: FrequencyWave(0.0, 0.0, 0.0))
    position: Tensor3D = field(default_factory=Tensor3D)
    velocity: Tensor3D = field(default_factory=Tensor3D)
    source_id: Optional[str] = None
    target_id: Optional[str] = None

    def step(self, dt: float = 0.1):
        self.position = self.position + (self.velocity * dt)
        self.payload = self.payload.step(dt)

    def to_dict(self):
        return {
            'id': self.id,
            'payload': self.payload.to_dict(),
            'position': self.position.to_dict(),
            'velocity': self.velocity.to_dict(),
            'source_id': self.source_id,
            'target_id': self.target_id
        }

@dataclass
class SoulTensor:
    """
    The unified physics object for Elysia's internal state.
    Combines Spatial Structure (Tensor3D) with Temporal Dynamics (FrequencyWave).
    This is the 'Coil' that stores and transmits 'Meaning'.
    """
    space: Tensor3D = field(default_factory=Tensor3D)
    wave: FrequencyWave = field(default_factory=lambda: FrequencyWave(0.0, 0.0, 0.0))

    # The 'Coil' factor: How tightly wound the energy is.
    # High spin = Concentrated, potential energy. Low spin = Radiating.
    spin: float = 0.0

    # Entanglement ID: If present, this tensor is synchronized with a SharedQuantumState.
    entanglement_id: Optional[str] = None

    def resonate(self, other: 'SoulTensor') -> 'SoulTensor':
        """
        The core interaction logic.
        Combines spatial alignment (Space) and wave interference (Time).
        """
        # 1. Spatial Alignment (How much do they overlap in meaning?)
        alignment = self.space.normalize().dot(other.space.normalize())
        # Alignment serves as a gate/multiplier for the wave interaction
        # If spatial concepts are orthogonal (0), waves can't fully interfere.
        coupling_coefficient = max(0.1, (alignment + 1.0) / 2.0) # Map -1..1 to 0..1

        # 2. Wave Interaction
        new_wave = self.wave.interact(other.wave)

        # Adjust wave amplitude based on spatial coupling
        new_wave.amplitude *= coupling_coefficient

        # 3. Space Merging
        # Simply adding vectors, but weighted by wave energy?
        # For now, simple vector addition.
        new_space = self.space + other.space

        # 4. Spin Interaction
        # Conservation of angular momentum metaphor
        new_spin = (self.spin + other.spin) / 2.0

        # 5. Entanglement Preservation
        # Interaction breaks entanglement unless specifically handled.
        # For now, resonance produces a NEW local state, breaking entanglement.
        # Use explicit 'entangle()' logic to re-establish.

        return SoulTensor(new_space, new_wave, new_spin)

    def to_dict(self):
        d = {
            'space': self.space.to_dict(),
            'wave': self.wave.to_dict(),
            'spin': self.spin
        }
        if self.entanglement_id:
            d['entanglement_id'] = self.entanglement_id
        return d

    @staticmethod
    def from_dict(data):
        if not data: return SoulTensor()
        space = Tensor3D.from_dict(data.get('space'))
        wave = FrequencyWave.from_dict(data.get('wave', {}))
        st = SoulTensor(space, wave, data.get('spin', 0.0))
        st.entanglement_id = data.get('entanglement_id')
        return st

@dataclass
class SharedQuantumState:
    """
    A container for a SoulTensor that is shared among multiple observers.
    This implements True Entanglement (Non-local shared state).
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tensor: SoulTensor = field(default_factory=lambda: SoulTensor())
    observers: List[str] = field(default_factory=list)

    def update(self, new_tensor: SoulTensor):
        self.tensor = new_tensor
