"""
Wave Tensor Calculus Module
===========================
"Mathematics of Resonance"
[GENESIS PROTOCOL V1.0]

This module implements the fundamental data structure for Elysia's 3rd Generation Computing:
The WaveTensor.

Unlike scalar numbers or standard vectors, a WaveTensor represents information as a superposition
of standing waves. It replaces boolean logic (True/False) with Harmonic Logic (Consonance/Dissonance).

Key Concepts:
- Superposition (Interference): $A + B$ is not arithmetic sum, but wave interference.
- Resonance (Dot Product): $A   B$ measures how much two waves "sing together" (Consonance).
- Phase Encoding: High-dimensional information is compressed into phase shifts.
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Optional

@dataclass
class WaveComponent:
    """A single frequency component of a thought/concept."""
    frequency: float  # Hz (The "Dimension" or "Identity")
    amplitude: float  # Magnitude (Importance/Energy)
    phase: float      # Radians (Relationship/Context)

    def to_complex(self) -> complex:
        """Convert to complex number representation for easy math."""
        return self.amplitude * np.exp(1j * self.phase)

class WaveTensor:
    """
    A multi-dimensional container of waves.
    Represents a complex thought, emotion, or data point in the Frequency Domain.
    """
    
    def __init__(self, name: str = "Anonymous Wave"):
        self.name = name
        # Vectorized Storage: Frequencies and their Complex Amplitudes
        self._frequencies = np.array([], dtype=np.float64)
        self._amplitudes = np.array([], dtype=np.complex128)
        
    @property
    def total_energy(self) -> float:
        """Returns total energy (sum of squared amplitudes). Vectorized."""
        if self._amplitudes.size == 0:
            return 0.0
        return np.sum(np.abs(self._amplitudes)**2)

    @property
    def active_frequencies(self) -> np.ndarray:
        return self._frequencies

    @property
    def dominant_frequency(self) -> float:
        """Returns the frequency with the highest amplitude magnitude."""
        if self._amplitudes.size == 0:
            return 0.0
        idx = np.argmax(np.abs(self._amplitudes))
        return self._frequencies[idx]

    def add_component(self, frequency: float, amplitude: float = 1.0, phase: float = 0.0):
        """Adds a single wave component. Merges if frequency already exists."""
        z = amplitude * np.exp(1j * phase)
        
        idx = np.searchsorted(self._frequencies, frequency)
        
        if idx < self._frequencies.size and self._frequencies[idx] == frequency:
            self._amplitudes[idx] += z
        else:
            self._frequencies = np.insert(self._frequencies, idx, frequency)
            self._amplitudes = np.insert(self._amplitudes, idx, z)

    def superpose(self, other: 'WaveTensor') -> 'WaveTensor':
        """
        [Superposition / Interference] - Vectorized implementation.
        """
        result = WaveTensor(f"Superposition({self.name}, {other.name})")
        
        # Merge frequencies using set union and sort
        all_freqs = np.unique(np.concatenate([self._frequencies, other._frequencies]))
        
        # Initialize zero-vectors for the union of frequencies
        v1 = np.zeros_like(all_freqs, dtype=np.complex128)
        v2 = np.zeros_like(all_freqs, dtype=np.complex128)
        
        # Map existing values to the new frequency grid
        v1[np.searchsorted(all_freqs, self._frequencies)] = self._amplitudes
        v2[np.searchsorted(all_freqs, other._frequencies)] = other._amplitudes
        
        result._frequencies = all_freqs
        result._amplitudes = v1 + v2
            
        return result

    def resonance(self, other: 'WaveTensor') -> float:
        """
        [Resonance / Consonance] - Fully Vectorized Dot Product.
        Returns a value between 0.0 (Dissonance) and 1.0 (Perfect Resonance).
        """
        return abs(self.interference(other))

    def interference(self, other: 'WaveTensor') -> float:
        """
        [Phase Interference / Epistemic Alignment]
        Returns a value between -1.0 (Destructive/Contradiction) and 1.0 (Constructive/Agreement).
        """
        if self._amplitudes.size == 0 or other._amplitudes.size == 0:
            return 0.0

        common_freqs = np.intersect1d(self._frequencies, other._frequencies)
        if common_freqs.size == 0:
            return 0.0
            
        z1 = self._amplitudes[np.searchsorted(self._frequencies, common_freqs)]
        z2 = other._amplitudes[np.searchsorted(other._frequencies, common_freqs)]
        
        dot_product = np.sum(z1 * np.conj(z2))
        energy_self = self.total_energy
        energy_other = other.total_energy
            
        alignment = (dot_product.real) / (math.sqrt(energy_self) * math.sqrt(energy_other))
        return float(alignment)

    def phase_shift(self, radians: float):
        """
        Rotates the entire tensor's phase. Vectorized.
        """
        rotator = np.exp(1j * radians)
        self._amplitudes *= rotator

    def normalize(self, target_energy: float = 1.0) -> 'WaveTensor':
        """
        Scales the tensor so that its total energy equals target_energy. Vectorized.
        """
        current_energy = self.total_energy
        if current_energy == 0:
            return self

        scale_factor = math.sqrt(target_energy / current_energy)
        self._amplitudes *= scale_factor

        return self

    def __repr__(self):
        components = self._frequencies.size
        energy = self.total_energy

        # Identify dominant frequency
        dom_freq = "None"
        if components > 0:
            dom_freq = f"{self.dominant_frequency:.1f}Hz"

        return f"<WaveTensor '{self.name}': E={energy:.2f}, Dom={dom_freq}, Components={components}>"

    # -- Standard Operators --
    
    def __add__(self, other):
        if isinstance(other, WaveTensor):
            return self.superpose(other)
        raise TypeError("Can only superpose WaveTensor with WaveTensor")

    def __mul__(self, scalar: Union[float, int]) -> 'WaveTensor':
        """Scalar multiplication (Scaling). Vectorized."""
        if isinstance(scalar, (float, int)):
            result = WaveTensor(f"{self.name} * {scalar}")
            result._frequencies = self._frequencies.copy()
            result._amplitudes = self._amplitudes * scalar
            return result
        raise TypeError("Can only multiply WaveTensor by scalar")

    def __matmul__(self, other):
        # Using @ operator for Resonance check
        if isinstance(other, WaveTensor):
            return self.resonance(other)
        raise TypeError("Can only check resonance with WaveTensor")

    def to_dict(self) -> dict:
        """Serializes the WaveTensor to a JSON-safe dictionary."""
        spectrum_data = []
        for i in range(self._frequencies.size):
            f = self._frequencies[i]
            z = self._amplitudes[i]
            spectrum_data.append([float(f), float(z.real), float(z.imag)])
        return {
            "name": self.name,
            "spectrum": spectrum_data
        }

    @staticmethod
    def from_dict(data: dict) -> 'WaveTensor':
        """Reconstructs a WaveTensor from a dictionary."""
        wt = WaveTensor(data.get("name", "Unknown Wave"))
        spectrum = data.get("spectrum", [])
        if spectrum:
            freqs = []
            vals = []
            for item in spectrum:
                f, r, i = item
                freqs.append(f)
                vals.append(complex(r, i))
            wt._frequencies = np.array(freqs)
            wt._amplitudes = np.array(vals)
        return wt

# --- Factory Methods ---

def create_harmonic_series(base_freq: float, harmonics: int = 4, decay: float = 0.5) -> WaveTensor:
    """Creates a rich, natural-sounding wave structure."""
    wt = WaveTensor(f"Harmonic({base_freq}Hz)")
    wt.add_component(base_freq, 1.0, 0.0)
    for i in range(1, harmonics + 1):
        freq = base_freq * (i + 1)
        amp = decay ** i
        wt.add_component(freq, amp, 0.0)
    return wt