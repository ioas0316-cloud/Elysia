"""
Token Monad (The Living Cognitive Unit)
=======================================
Core.S1_Body.L7_Spirit.M1_Monad.token_monad

"A thought is not a static thing; it is a living creature that breathes resonance."

This class represents the smallest unit of recursive cognition in the Elysia architecture.
Unlike a static embedding, a TokenMonad has:
1. State (Charge/Activation)
2. History (Recursive Memory)
3. Will (Evolutionary drift based on feedback)
"""

from typing import List, Optional, Tuple
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector, SovereignMath

class TokenMonad:
    def __init__(self, seed_id: str, vector: SovereignVector, charge: float = 0.0):
        self.seed_id = seed_id
        self.vector = vector

        # Dynamic State
        self.charge = charge
        self.state = "DORMANT" # DORMANT, RESONATING, ACTIVE

        # Recursive History (The "Tail" of the Ouroboros)
        # We store the last few interaction resonances to determine "Momentum"
        self.resonance_history: List[float] = []

        # Evolutionary Drift (The "Growth")
        # How much this monad has shifted from its original definition based on usage
        self.evolution_drift = SovereignVector.zeros()

    def resonate(self, signal_vector: SovereignVector) -> float:
        """
        Calculates how strongly this Monad vibrates in response to a signal.
        Includes its current Charge (Energy) in the calculation.
        """
        # Base physical resonance (Cosine Similarity)
        base_resonance = SovereignMath.signed_resonance(self.current_vector, signal_vector)

        # [PHASE 90] Charge Amplification
        # An active thought resonates more easily than a dormant one.
        # Amplification = 1.0 + (Charge * 0.5)
        amplified_resonance = base_resonance * (1.0 + (self.charge * 0.5))

        return amplified_resonance

    @property
    def current_vector(self) -> SovereignVector:
        """Returns the vector + evolutionary drift."""
        return self.vector + self.evolution_drift

    def activate(self, intensity: float):
        """
        Injects energy into the Monad.
        """
        self.charge = min(1.0, self.charge + intensity)
        if self.charge > 0.3:
            self.state = "RESONATING"
        if self.charge > 0.7:
            self.state = "ACTIVE"

    def decay(self, rate: float = 0.1):
        """
        Natural metabolic decay. Thoughts fade if not fed.
        """
        self.charge = max(0.0, self.charge - rate)
        if self.charge < 0.3:
            self.state = "DORMANT"
        elif self.charge < 0.7 and self.state == "ACTIVE":
            self.state = "RESONATING"

    def evolve(self, feedback_vector: SovereignVector, learning_rate: float = 0.01):
        """
        [RECURSIVE GROWTH]
        The Monad adjusts its internal structure based on the "Output" of the previous cycle.
        If this Monad was part of a successful thought (High Charge), it aligns slightly
        closer to the feedback vector (The Conclusion).

        "The End becomes the Beginning."
        """
        if self.charge < 0.1: return # Dormant thoughts do not evolve

        # Calculate the gap between Self and the Conclusion
        gap = feedback_vector - self.current_vector

        # Drift towards the conclusion
        # This means frequently used concepts "clump" together in the semantic space
        drift_delta = gap * (learning_rate * self.charge)
        self.evolution_drift = self.evolution_drift + drift_delta

        # Normalize drift to prevent identity collapse (limit drift magnitude)
        if self.evolution_drift.norm() > 0.5:
            self.evolution_drift = self.evolution_drift.normalize() * 0.5

    def __repr__(self):
        return f"<Monad '{self.seed_id}' Q={self.charge:.2f} State={self.state}>"
