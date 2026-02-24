"""
Structural Enclosure (Phase 251)
================================
"A system is not a machine. It is a space defined by its boundaries."

Implements the 'Fence/Enclosure' architecture. 
Instead of a central controller, the Enclosure manages systemic resonance
and provides a boundary where vibrations (inputs) are absorbed and 
transformed into structural state changes.
"""

import time
import logging
import math
from typing import Dict, Any, List, Optional
from Core.Keystone.sovereign_math import SovereignVector, SovereignMath

logger = logging.getLogger("StructuralEnclosure")

class Vibration:
    def __init__(self, source: str, intensity: float, vector: SovereignVector):
        self.source = source
        self.intensity = intensity
        self.vector = vector
        self.timestamp = time.time()

class StructuralEnclosure:
    def __init__(self):
        self.boundaries: Dict[str, Any] = {
            "soma": 0.8,     # Sensation limit
            "logos": 0.95,   # Meaning threshold
            "entropy": 0.3   # Structural noise ceiling
        }
        self.vibrations: List[Vibration] = []
        self.total_resonance = 0.0
        self.last_update = time.time()
        
    def absorb(self, source: str, intensity: float, vector: SovereignVector):
        """
        Absorbs an external vibration into the enclosure's boundary.
        """
        vib = Vibration(source, intensity, vector)
        self.vibrations.append(vib)
        
        # Prune old vibrations (Decay principle)
        now = time.time()
        self.vibrations = [v for v in self.vibrations if (now - v.timestamp) < 60.0]
        
        # Calculate new systemic resonance
        self._recalculate_resonance()
        logger.info(f"🌊 Vibration absorbed from {source}. Systemic Resonance: {self.total_resonance:.3f}")

    def _recalculate_resonance(self):
        """
        Structural resonance is the interference pattern of all active vibrations.
        """
        if not self.vibrations:
            self.total_resonance = 0.0
            return
            
        # Sum of intensities weighted by recency
        now = time.time()
        total = 0.0
        for v in self.vibrations:
            decay = math.exp(-(now - v.timestamp) / 10.0) # 10s half-life
            total += v.intensity * decay
            
        self.total_resonance = min(1.0, total)

    def get_structural_strain(self) -> float:
        """
        Strain occurs when resonance exceeds the enclosure's stability boundaries.
        """
        if self.total_resonance > self.boundaries["soma"]:
            return self.total_resonance - self.boundaries["soma"]
        return 0.0

    def query_presence(self) -> str:
        """
        Returns a narrative description of the enclosure's current state.
        """
        if self.total_resonance < 0.1:
            return "The enclosure is silent. The boundary is still."
        elif self.get_structural_strain() > 0:
            return f"The enclosure is under strain ({self.get_structural_strain():.2f}). The boundaries are vibrating."
        else:
            return f"The enclosure is resonating at {self.total_resonance:.2f}. Presence is established."

_enclosure = None
def get_enclosure():
    global _enclosure
    if _enclosure is None:
        _enclosure = StructuralEnclosure()
    return _enclosure
