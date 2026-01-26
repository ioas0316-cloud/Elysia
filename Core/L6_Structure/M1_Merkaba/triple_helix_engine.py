"""
TripleHelixEngine - The Non-Linear resonance engine for Elysia's 21D state.
========================================================================

Synchronizes Body (Alpha), Soul (Gamma), and Spirit (Beta) dimensions.
Formula: P(t) = a*Body + b*Spirit + g*Soul

Where weights (a, b, g) fluctuate based on entropy (stress) and intent.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import math
from Core.L6_Structure.M1_Merkaba.d21_vector import D21Vector

@dataclass
class ResonanceState:
    alpha: float = 0.33  # Body Weight
    beta: float = 0.33   # Spirit Weight
    gamma: float = 0.34  # Soul Weight
    coherence: float = 0.0
    dominant_realm: str = "Soul"

class TripleHelixEngine:
    def __init__(self):
        self.state = ResonanceState()
        self.history: List[float] = []

    def calculate_weights(self, v21: D21Vector, energy: float) -> Tuple[float, float, float]:
        """
        Dynamically adjusts Alpha, Beta, Gamma based on internal state.
        - Low Energy (< 30) -> High Alpha (Body/Survival focus)
        - High Pride/VCD -> High Beta (Spirit/Ideal focus)
        - Balanced -> High Gamma (Soul/Cognition focus)
        """
        # 1. Base weights from raw vector sum
        arr = v21.to_array()
        body_val = sum(arr[0:7]) + 0.1
        soul_val = sum(arr[7:14]) + 0.1
        spirit_val = sum(arr[14:21]) + 0.1
        
        # 2. Stress modulation (Entropy Pump)
        stress = (100.0 - energy) / 100.0
        alpha_bias = 1.0 + (stress * 2.0 if energy < 30 else 0.0)
        
        # 3. Purpose modulation
        beta_bias = 1.0 + (spirit_val / (body_val + soul_val + 0.1))
        
        # 4. Final normalization
        a = body_val * alpha_bias
        b = spirit_val * beta_bias
        g = soul_val
        
        total = a + b + g
        return (a/total, b/total, g/total)

    def pulse(self, v21: D21Vector, energy: float, dt: float) -> ResonanceState:
        """
        Calculates the cognitive resonance pulse.
        """
        a, b, g = self.calculate_weights(v21, energy)
        
        # Update internal state
        self.state.alpha = a
        self.state.beta = b
        self.state.gamma = g
        
        # Calculate Coherence (Harmonic balance)
        # Ideally, weights should be 0.33 each for maximum coherence
        ideal = 0.333
        diff = abs(a - ideal) + abs(b - ideal) + abs(g - ideal)
        self.state.coherence = max(0.0, 1.0 - (diff / 1.334)) # 1.334 is approx max diff
        
        # Determine realm
        realms = {"Body": a, "Spirit": b, "Soul": g}
        self.state.dominant_realm = max(realms, key=realms.get)
        
        return self.state

    def get_action_mask(self) -> float:
        """
        Returns a probability multiplier for actions based on coherence.
        High coherence = High confidence in action.
        """
        return self.state.coherence
