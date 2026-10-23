import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class IonState:
    phase: complex
    permittivity: float
    inertia: float
    bit_density: float

class Rotor:
    """
    [Rotor: The Atom of Dielectric Physics]
    Converts raw bit streams into complex-phase 'ions'.
    Bit density (1s vs 0s) determines the local permittivity and inertia of the data.
    """
    def __init__(self, history_size: int = 64):
        self.history_size = history_size
        self.bit_history: List[int] = []
        self.current_inertia = 1.0

    def process_bits(self, bit_stream: bytes) -> IonState:
        """
        Processes a byte stream, calculating bit density and projecting it into a complex phase.
        """
        bits = []
        for byte in bit_stream:
            for i in range(8):
                bits.append((byte >> (7-i)) & 1)

        # Update history
        self.bit_history.extend(bits)
        if len(self.bit_history) > self.history_size:
            self.bit_history = self.bit_history[-self.history_size:]

        # Calculate Bit Density (1s as positive ions, 0s as negative ions)
        ones = sum(bits)
        zeros = len(bits) - ones
        total = len(bits) if len(bits) > 0 else 1

        bit_density = (ones - zeros) / total # Range: -1.0 to 1.0

        # Calculate Permittivity: High density (of either) increases resistance/permittivity
        # Absolute density of information
        abs_density = abs(bit_density)
        permittivity = 1.0 + abs_density * 5.0 # Information 'thickness'

        # Calculate Phase: The angle on the complex plane representing the ion's polarity
        # 1s (Positive) -> 0 rad, 0s (Negative) -> PI rad?
        # Or better: Use the bit_density to define an angle.
        angle = bit_density * np.pi
        phase = np.exp(1j * angle)

        # Update Inertia: Historical accumulation of density
        history_density = sum(self.bit_history) / len(self.bit_history) if self.bit_history else 0.5
        self.current_inertia = 1.0 + history_density * 2.0

        return IonState(
            phase=phase,
            permittivity=permittivity,
            inertia=self.current_inertia,
            bit_density=bit_density
        )

    def __repr__(self):
        return f"Rotor(Inertia: {self.current_inertia:.2f})"
