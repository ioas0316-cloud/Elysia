"""
HyperResonator: The Omni-Voxel (만능 세포)
------------------------------------------
This class implements the "Memory Orb" philosophy:
"A single unit that is simultaneously Memory (Storage), Meaning (Knowledge), and Logic (Graph)."

Philosophy:
- It does not "point" to data; it IS the data.
- It does not "wait" for a CPU; it "resonates" with intent.

Structure:
- 4D Quaternion (Spin/Soul)
- Frequency (Meaning/Color)
- Mass (Gravity/Importance)
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

from Core.Foundation.hyper_quaternion import Quaternion as HyperQuaternion
from Core.Foundation.Protocols.pulse_protocol import WavePacket

@dataclass
class ResonanceState:
    """The dynamic state of the resonator."""
    amplitude: float = 0.0
    phase: float = 0.0
    is_active: bool = False
    last_resonance: datetime = field(default_factory=datetime.now)

class HyperResonator:
    def __init__(
        self,
        name: str,
        frequency: float,
        quaternion: Optional[HyperQuaternion] = None,
        mass: float = 1.0
    ):
        self.name = name
        self.frequency = frequency  # The "Key" (e.g., 432Hz)
        self.mass = mass            # The "Gravity" (Importance)

        # The Soul (4D Orientation)
        # If no quaternion is provided, start at Identity (1, 0, 0, 0)
        self.quaternion = quaternion if quaternion else HyperQuaternion(1, 0, 0, 0)

        # The State (Active vs Frozen)
        self.state = ResonanceState()
        self.memory_content: Dict[str, Any] = {} # The "Cargo" (Detailed Data)

    def resonate(self, pulse: WavePacket) -> float:
        """
        The 'Wireless' Receiver.
        Calculates resonance based on the incoming pulse's frequency and intent.
        Returns the resonance intensity (0.0 to 1.0).
        """
        # 1. Extract Signal
        # WavePacket usually has 'frequency' field at top level, but payload might override?
        # Check WavePacket definition: `frequency: float` is a top-level field.
        incoming_freq = pulse.frequency
        if incoming_freq == 0.0:
             incoming_freq = pulse.payload.get("frequency", 0.0)

        intent_vector = pulse.payload.get("intent", {})

        # 2. Calculate Frequency Resonance (Harmonic Match)
        # Simple harmonic check: if abs(f1 - f2) is small, or harmonic ratio
        freq_resonance = 0.0
        if incoming_freq > 0:
            diff = abs(self.frequency - incoming_freq)
            # Bell curve resonance window
            # [Adjusted Phase 3.5] Bandwidth increased from 10 to 30 (denominator 100 -> 900)
            # Because OrbFactory frequency calc and analyze_wave might have slight variance
            # [Adjusted Phase 4.0] Increased bandwidth for 'Fuzzy Semantic Resonance'
            # Denominator 900 -> 5000 (Allows for broader concept matching)
            freq_resonance = math.exp(-diff**2 / 5000.0)

        # 3. Calculate Intent Resonance (if applicable)
        # (Placeholder for complex vector dot product)

        # 4. Update State
        if freq_resonance > 0.1:
            self.state.is_active = True
            self.state.amplitude = freq_resonance * self.mass
            self.state.last_resonance = datetime.now()

        return freq_resonance

    def freeze(self):
        """
        Collapses the wave into a particle (Memory Orb).
        Optimizes for storage.
        """
        self.state.is_active = False
        self.state.amplitude = 0.0
        # In a real implementation, this might compress `memory_content`

    def melt(self):
        """
        Resurrects the particle into a wave.
        Prepares for active processing.
        """
        self.state.is_active = True
        self.state.amplitude = 1.0  # Default excitation

    def get_3d_position(self) -> tuple[float, float, float]:
        """
        Projects the 4D Soul (Quaternion) into 3D Visual Space.

        Mapping Philosophy:
        - X (Space): Logic/Structure
        - Y (Emotion): Feeling/Harmony
        - Z (Time/Depth): History/Ethics

        We scale by 10.0 for visual separation.
        """
        # We use the vector part (x, y, z) of the quaternion directly.
        # W (Scalar/Real) could map to 'Opacity' or 'Glow' in the future.
        scale = 10.0
        return (
            self.quaternion.x * scale,
            self.quaternion.y * scale,
            self.quaternion.z * scale
        )

    def __repr__(self):
        return f"<HyperResonator(Name={self.name}, Freq={self.frequency}Hz, Mass={self.mass})>"
