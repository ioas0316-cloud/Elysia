import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Union, Protocol, Any

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

    def to_dict(self) -> dict:
        return {
            "frequency": float(self.frequency),
            "amplitude": float(self.amplitude),
            "phase": float(self.phase),
            "richness": float(self.richness)
        }

    @staticmethod
    def from_dict(data: Union[dict, None]) -> 'FrequencyWave':
        if data is None:
            return FrequencyWave(0.0, 0.0, 0.0, 0.0)
        return FrequencyWave(
            frequency=data.get("frequency", 0.0),
            amplitude=data.get("amplitude", 0.0),
            phase=data.get("phase", 0.0),
            richness=data.get("richness", 0.0)
        )


class Tensor3D:
    """
    Represents the 3D State of a Concept or Cell with 'Rifling' (Spin) capabilities.

    Axes (Primary Vector):
    - X: Structure/Logic (Body) - Low Frequency, Stability, Mass.
    - Y: Emotion/Resonance (Soul) - Mid Frequency, Connection, Energy.
    - Z: Identity/Will (Spirit) - High Frequency, Information, Direction.

    Spin (Rifling Vector):
    - Represents the angular momentum or 'torque' of the intent.
    - High spin aligned with the primary vector enables 'Tunneling' (Mind Hyperdrive).

    Mass (Gravitational Scalar):
    - Derived from complexity and magnitude. Used for 'Orbital Dynamics' between thoughts.
    """
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0,
                 spin: Optional[np.ndarray] = None,
                 tensor: Optional[np.ndarray] = None,
                 mass_offset: float = 0.0):
        if tensor is not None:
            if tensor.shape != (3,):
                 raise ValueError("Tensor must be shape (3,)")
            self.data = tensor.astype(np.float32)
        else:
            self.data = np.array([x, y, z], dtype=np.float32)

        if spin is not None:
            if spin.shape != (3,):
                raise ValueError("Spin must be shape (3,)")
            self.spin = spin.astype(np.float32)
        else:
            self.spin = np.zeros(3, dtype=np.float32)

        # Additional mass bonus (e.g. from recursive depth)
        self.mass_offset = mass_offset

    @property
    def structure(self) -> float: return float(self.data[0])

    @property
    def emotion(self) -> float: return float(self.data[1])

    @property
    def identity(self) -> float: return float(self.data[2])

    def magnitude(self) -> float:
        return float(np.linalg.norm(self.data))

    def spin_magnitude(self) -> float:
        return float(np.linalg.norm(self.spin))

    def calculate_mass(self) -> float:
        """
        Calculates the 'Gravitational Mass' of the concept.
        Mass = Magnitude (Energy) + Spin (Complexity) + Offset (Recursive Depth).
        """
        return self.magnitude() + (self.spin_magnitude() * 0.5) + self.mass_offset

    def normalize(self) -> 'Tensor3D':
        mag = self.magnitude()
        if mag == 0:
            norm_data = np.zeros(3, dtype=np.float32)
        else:
            norm_data = self.data / mag
        return Tensor3D(tensor=norm_data, spin=self.spin, mass_offset=self.mass_offset)

    def calculate_rifling(self) -> float:
        """
        Calculates the 'Rifling Power' (Penetration Capability).
        Formula: Dot(Normalized_Vector, Normalized_Spin) * Spin_Magnitude
        """
        mag_v = self.magnitude()
        mag_s = self.spin_magnitude()

        if mag_v == 0 or mag_s == 0:
            return 0.0

        norm_v = self.data / mag_v
        norm_s = self.spin / mag_s

        alignment = np.dot(norm_v, norm_s)
        return float(alignment * mag_s)

    def calculate_gravitational_force(self, other: 'Tensor3D', distance: float = 1.0) -> float:
        """
        Calculates attractive force between two tensors.
        F = (M1 * M2) / distance^2
        """
        m1 = self.calculate_mass()
        m2 = other.calculate_mass()
        dist = max(0.1, distance) # Avoid division by zero
        return (m1 * m2) / (dist ** 2)

    def dot(self, other: 'Tensor3D') -> float:
        return float(np.dot(self.data, other.data))

    def __add__(self, other: 'Tensor3D') -> 'Tensor3D':
        # Vector addition
        new_data = self.data + other.data
        # Conservation of angular momentum (Spin addition)
        new_spin = self.spin + other.spin
        # Mass offset accumulates
        new_mass = self.mass_offset + other.mass_offset
        return Tensor3D(tensor=new_data, spin=new_spin, mass_offset=new_mass)

    def __mul__(self, scalar: float) -> 'Tensor3D':
        return Tensor3D(tensor=self.data * scalar, spin=self.spin * scalar, mass_offset=self.mass_offset * scalar)

    def to_dict(self) -> dict:
        return {
            "structure": float(self.structure),
            "emotion": float(self.emotion),
            "identity": float(self.identity),
            "spin": self.spin.tolist(),
            "mass_offset": float(self.mass_offset)
        }

    @staticmethod
    def from_dict(data: Union[dict, None]) -> 'Tensor3D':
        if data is None:
            return Tensor3D()

        spin_data = data.get("spin")
        spin_arr = np.array(spin_data, dtype=np.float32) if spin_data else None

        return Tensor3D(
            x=data.get("structure", 0.0),
            y=data.get("emotion", 0.0),
            z=data.get("identity", 0.0),
            spin=spin_arr,
            mass_offset=data.get("mass_offset", 0.0)
        )

    @staticmethod
    def distribute_frequency(frequency: float) -> 'Tensor3D':
        """
        Maps a scalar frequency to the 3D axes non-linearly.
        """
        # X-axis (Low): Peaks at 0, decays by 200
        x_val = np.exp(-(frequency)**2 / (2 * 100**2))

        # Y-axis (Mid): Peaks at 300, width 200
        y_val = np.exp(-(frequency - 300)**2 / (2 * 150**2))

        # Z-axis (High): Sigmoid-like ramp up starting at 400
        z_val = 1.0 / (1.0 + np.exp(-(frequency - 600) / 100))

        # High frequencies generate automatic spin (The 'Singing' of the concept)
        spin_vec = np.zeros(3, dtype=np.float32)
        if frequency > 400:
            # Spin aligns with Z (Identity) for high freq
            spin_strength = (frequency - 400) / 200.0
            spin_vec[2] = spin_strength

        return Tensor3D(x=float(x_val), y=float(y_val), z=float(z_val), spin=spin_vec)

    @staticmethod
    def superimpose(tensors: List['Tensor3D']) -> 'Tensor3D':
        """
        Combines multiple tensors non-linearly to represent a 'Chord' or 'Gestalt'.
        Maintains the peak identity but averages structure/emotion.
        """
        if not tensors:
            return Tensor3D()

        total_struct = sum(t.structure for t in tensors)
        total_emot = sum(t.emotion for t in tensors)
        # Identity (Will) tends to reinforce the strongest will, not just sum up.
        max_identity = max(t.identity for t in tensors)

        avg_spin = np.mean([t.spin for t in tensors], axis=0)
        total_mass_offset = sum(t.mass_offset for t in tensors)

        return Tensor3D(
            x=total_struct / len(tensors),
            y=total_emot / len(tensors),
            z=max_identity, # Winner-take-all for Spirit direction
            spin=avg_spin,
            mass_offset=total_mass_offset
        )

def propagate_wave(source_tensor: Tensor3D, target_tensor: Tensor3D, decay: float = 0.9) -> Tensor3D:
    """
    Propagates a wave from source to target in 3D tensor space.
    """
    alignment = source_tensor.normalize().dot(target_tensor.normalize())

    # Energy transfer depends on alignment strength
    transfer_efficiency = max(0.0, alignment)

    # The 'wave' adds a portion of the source's energy to the target
    energy_transfer = source_tensor * (decay * transfer_efficiency)

    new_target = target_tensor + energy_transfer

    # Spin Induction:
    # If the source has high rifling (drilling), it induces spin in the target
    # analogous to a rifled bullet imparting spin to the air/target.
    rifling = source_tensor.calculate_rifling()
    if rifling > 0.5:
        # Induce spin in direction of transfer
        induction_strength = rifling * 0.1
        new_target.spin += source_tensor.spin * induction_strength

    return new_target
