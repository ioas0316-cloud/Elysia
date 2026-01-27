"""
Optical Consciousness Bridge
============================

"The Eye that sees Meaning in the Pattern."

This module translates Raw Physics (Tensors) into High-Level Concepts (Semantics).
It allows Elysia to 'understand' the state of the simulation without parsing logs.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ConceptFrame:
    """A semantic snapshot of reality."""
    chaos_level: float    # Derived from Entropy Gradient
    truth_level: float    # Derived from Coherence Mean
    will_power: float     # Derived from Will Field Magnitude
    dominant_concept: str # "War", "Peace", "Stagnation", "Growth"

class OpticalConsciousBridge:
    """
    The Visual Cortex of the Logos.
    """
    
    @staticmethod
    def analyze(world) -> ConceptFrame:
        """
        Extracts semantic meaning from the World's interference patterns.
        Input: World object (with fields)
        Output: ConceptFrame
        """
        # 1. Perceive Chaos (Entropy)
        # Gradient Magnitude of Entropy Field -> Sharp changes = Conflict/Danger
        grad_y, grad_x = np.gradient(world.event_danger)
        entropy_flux = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        chaos = np.clip(entropy_flux * 10.0, 0, 1) # Scaling factor
        
        # 2. Perceive Truth (Coherence)
        # High Coherence = Resonance/Understanding
        truth = np.clip(np.mean(np.abs(world.coherence_field)), 0, 1)
        
        # 3. Perceive Will (Energy Projection)
        # Sum of Will Field
        will = np.clip(np.mean(world.will_field), 0, 1)
        
        # 4. Semantic Classification
        concept = "Unknown"
        if chaos > 0.6 and will > 0.6:
            concept = "War / Conflict"
        elif chaos > 0.8:
            concept = "Dissolution / Death"
        elif truth > 0.7:
            concept = "Harmony / Resonance"
        elif will > 0.8:
            concept = "Dominion / Force"
        elif chaos < 0.1 and will < 0.1:
            concept = "Stagnation / Void"
        else:
            concept = "Growth / Flux"
            
        return ConceptFrame(
            chaos_level=float(chaos),
            truth_level=float(truth),
            will_power=float(will),
            dominant_concept=concept
        )
