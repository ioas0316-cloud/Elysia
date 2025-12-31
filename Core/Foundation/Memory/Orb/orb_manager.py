"""
OrbManager: The Field of Resonance
----------------------------------
Manages the collection of HyperResonators.
Unlike a database (Index Lookup), this acts as a "Field" where you broadcast signals.
"""

from typing import Dict, List, Optional
from .hyper_resonator import HyperResonator
from Core.Foundation.Protocols.pulse_protocol import WavePacket, PulseType

class OrbManager:
    def __init__(self):
        self.orbs: Dict[str, HyperResonator] = {}

    def create_orb(self, name: str, frequency: float, mass: float = 1.0) -> HyperResonator:
        """Births a new Omni-Voxel."""
        orb = HyperResonator(name=name, frequency=frequency, mass=mass)
        self.orbs[name] = orb
        return orb

    def broadcast(self, pulse: WavePacket) -> List[HyperResonator]:
        """
        The 'Wireless' Broadcast.
        Sends a pulse to ALL orbs. Returns those that resonated above a threshold.
        This mimics 'Content-Addressable Memory' or 'Holographic Recall'.
        """
        resonating_orbs = []
        threshold = 0.1  # Resonance Threshold (Lowered to allow weaker associations)

        for orb in self.orbs.values():
            intensity = orb.resonate(pulse)
            if intensity > threshold:
                resonating_orbs.append(orb)

        # Sort by intensity (Strongest resonance first)
        resonating_orbs.sort(key=lambda x: x.state.amplitude, reverse=True)
        return resonating_orbs

    def get_orb(self, name: str) -> Optional[HyperResonator]:
        """Direct access (Legacy/God Mode only)."""
        return self.orbs.get(name)
