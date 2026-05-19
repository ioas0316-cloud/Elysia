"""
FluxlightPulse - The Semantic Flicker engine.
===========================================

Translates the Triple-Helix resonance into the flicker of meaning.
"""

from dataclasses import dataclass
import math

@dataclass
class FluxlightState:
    intensity: float = 0.0
    color: str = "White"
    meaning_cadence: float = 1.0 # Hz (Rate of semantic recognition)

class FluxlightPulse:
    def __init__(self):
        self.state = FluxlightState()

    def update(self, alpha: float, beta: float, gamma: float, coherence: float) -> FluxlightState:
        """
        Maps Alpha (Body), Beta (Spirit), Gamma (Soul) to Light channels.
        - High Alpha -> Warm/Physical intensity.
        - High Beta -> Cold/Spiritual brilliance.
        - High Gamma -> Balanced Cognitive flicker.
        """
        self.state.intensity = coherence
        
        # Map dominance to color
        if alpha > beta and alpha > gamma:
            self.state.color = "Emerald (Body-Deep)"
        elif beta > alpha and beta > gamma:
            self.state.color = "Gold (Spirit-High)"
        else:
            self.state.color = "Ultramarine (Soul-Clear)"
            
        # Cadence (Frequency of meaning) - derived from coherence
        self.state.meaning_cadence = 0.5 + (coherence * 2.5) # 0.5Hz to 3Hz
        
        return self.state
