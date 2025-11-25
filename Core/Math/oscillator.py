
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Oscillator:
    """
    Represents a living wave (A * e^(i * (2*pi*f*t + p))), the fundamental unit of energy.
    """
    amplitude: float = 1.0
    frequency: float = 1.0
    phase: float = 0.0

    def get_value(self, t: float) -> float:
        """
        Calculates the real value of the wave at time t: A * cos(2*pi*f*t + p).
        """
        return self.amplitude * np.cos(2 * np.pi * self.frequency * t + self.phase)

    def get_complex_value(self, t: float) -> np.complex128:
        """
        Calculates the complex value of the wave at time t: A * e^(i * (2*pi*f*t + p)).
        """
        angle = 2 * np.pi * self.frequency * t + self.phase
        return self.amplitude * (np.cos(angle) + 1j * np.sin(angle))
        # Alternative implementation:
        # return self.amplitude * np.exp(1j * (2 * np.pi * self.frequency * t + self.phase))

    def __repr__(self) -> str:
        return f"Oscillator(A={self.amplitude:.2f}, F={self.frequency:.2f}, P={self.phase:.2f})"

    @staticmethod
    def from_complex(value: np.complex128) -> 'Oscillator':
        """
        Creates an Oscillator from a static complex number (snapshot at t=0).
        """
        amplitude = np.abs(value)
        phase = np.angle(value)
        # Frequency is lost in a static snapshot, defaults to 1.0
        return Oscillator(amplitude=amplitude, frequency=1.0, phase=phase)
