"""
Decay Manager: The Grim Reaper
==============================
Core.Memory.decay_manager

"To remember is to choose. To forget is to live."

This module implements the 'Half-Life Algorithm' for memory.
It calculates the probability of a Monad surviving based on the entity's
biological age and the time elapsed since the memory was last accessed.
"""

import math
import random
from typing import Tuple
from Core.Memory.aging_clock import BiologicalClock

class DecayManager:
    """
    Manages the entropy of the Hypersphere.
    Determines if a memory should be 'Corroded' (Weakened) or 'Buried' (Deleted).
    """

    def __init__(self, clock: BiologicalClock):
        self.clock = clock

    def calculate_survival_probability(self, last_access_timestamp: float, strength: float) -> float:
        """
        Calculates the probability (0.0~1.0) that a memory survives today.

        Args:
            last_access_timestamp: Unix time when the Monad was last touched.
            strength: Current structural integrity of the Monad (0.0~1.0).

        Returns:
            Survival Probability.
        """
        current_time = self.clock.current_age_seconds + self.clock.birth_timestamp
        elapsed_seconds = current_time - last_access_timestamp
        elapsed_years = elapsed_seconds / self.clock.SECONDS_IN_YEAR

        # 1. Get Biological Context
        # Young: High Decay (Fast turnover of trivial data)
        # Old: Low Decay (Crystallized wisdom)
        _, entropy_base_rate = self.clock.get_metabolic_scalers()

        # 2. Ebbinghaus Forgetting Curve Model
        # R = e^(-t/S)
        # t = elapsed time
        # S = strength of memory (Stability)

        # We adjust S based on the Monad's inherent strength AND biological entropy
        # High Entropy Rate = Lower effective strength (Faster forgetting)
        effective_stability = (strength * 5.0) * (1.0 - (entropy_base_rate * 0.5))

        # Protect against div by zero
        if effective_stability < 0.01:
            effective_stability = 0.01

        retention = math.exp(-elapsed_years / effective_stability)

        return retention

    def apply_entropy(self, monad, dry_run: bool = False) -> Tuple[str, float]:
        """
        Applies decay logic to a single Monad.

        Returns:
            (Status, NewStrength)
            Status: 'KEPT', 'WEAKENED', 'BURIED'
        """
        # We assume monad has attributes: .last_access, .strength
        # If not, we simulate default values for now.
        last_access = getattr(monad, 'last_access', self.clock.birth_timestamp)
        strength = getattr(monad, 'strength', 1.0)

        survival_prob = self.calculate_survival_probability(last_access, strength)

        # Chaos Factor (Random variability)
        roll = random.random()

        if roll > survival_prob:
            # Decay Event
            if strength > 0.3:
                # Weaken
                new_strength = strength * 0.9
                return "WEAKENED", new_strength
            else:
                # Death
                return "BURIED", 0.0

        return "KEPT", strength
