"""
Phase Collector (State Container)
=================================
Core.Monad.phase_collector

"I am the vessel. The Law moves me."

The Phase Collector is now a lightweight state container.
It delegates all logic to the MerkabaCore.
"""

import hashlib
from typing import Dict, Any, Optional
from Core.Monad.merkaba_core import MerkabaCore, StateVector

class PhaseCollector:
    def __init__(self, collector_id: str, orbit_slot: int = 0):
        self.id = collector_id
        # The State Vector
        self.state: StateVector = {
            'phase': 0.0,
            'velocity': 0.0,
            'energy': 0.0,
            'resonance': 0.0
        }

    def update(self, input_phase: float):
        """
        The Heartbeat.
        Passes current state + input to the Law, receives next state.
        """
        self.state = MerkabaCore.apply_law(self.state, input_phase, depth=0)

    def absorb_radiance(self, radiant_data: str) -> float:
        """
        Boundary Interface: Text -> Phase -> Update.
        """
        # 1. Convert at Boundary
        input_phase = self._radiance_to_phase(radiant_data)

        # 2. Apply Law
        self.update(input_phase)

        return self.state['energy']

    def _radiance_to_phase(self, data: str) -> float:
        """Deterministic Hash to Phase."""
        seed_str = f"{data}_SPECTRA"
        hash_bytes = hashlib.sha256(seed_str.encode()).digest()
        return int.from_bytes(hash_bytes[:2], 'big') % 360.0

    def get_phase(self) -> float:
        return self.state['phase']

    def get_state(self) -> StateVector:
        return self.state

    def discharge(self) -> Dict[str, Any]:
        """Returns energy and resets partial accumulation (if needed)."""
        # In this recursive model, energy is continuous.
        # But for reporting we might want to 'read' it.
        return {
            "id": self.id,
            "phase": self.state['phase'],
            "energy": self.state['energy'],
            "resonance": self.state['resonance']
        }
