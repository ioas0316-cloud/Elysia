"""
Autonomous Transducer (The Experiential Bridge)
===============================================
"A word is a pointer to a feeling, not a box for a definition."

This module links symbolic language directly to the system's ACTIVE 21D state.
It implements 'Experiential Grounding' by snapshotting the physical manifold.
"""

from typing import Optional, Callable, Any, List
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignMath, SovereignVector

class AutonomousTransducer:
    def __init__(self, state_provider: Callable[[], Any]):
        """
        Args:
            state_provider: A function that returns the current active 21D state.
        """
        self.state_provider = state_provider

    def transduce_state(self) -> SovereignVector:
        """
        [PHASE 110] Captures the current active resonance as a SovereignVector.
        """
        v21 = self.state_provider()
        if isinstance(v21, SovereignVector):
            return v21
        # Handle legacy D21Vector or other types
        data = v21.data if hasattr(v21, 'data') else (v21.to_array() if hasattr(v21, 'to_array') else list(v21))
        return SovereignVector(data)

    def capture_experience(self) -> List[float]:
        """
        Compatibility method: Snapshots the current active resonance as a list.
        """
        return self.transduce_state().tolist()

    def bridge_symbol(self, symbol: str) -> SovereignVector:
        """
        Evokes the physical resonance for a given symbol.
        In this autonomous mode, it returns the current state if prompted.
        """
        return self.transduce_state()

    @staticmethod
    def calculate_resonance(v1: SovereignVector, v2: SovereignVector) -> float:
        """Cosine similarity between two state vectors."""
        return v1.resonance_score(v2)
