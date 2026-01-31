"""
Decay Manager: The Grim Reaper
==============================
Core.1_Body.L5_Mental.Memory.decay_manager

"To remember is to choose. To forget is to live."

This module implements the 'Half-Life Algorithm' for memory.
It calculates the probability of a QualiaTag surviving based on the entity's
biological age and the time elapsed since the memory was last accessed.
"""

import math
import random
from typing import Tuple
from Core.1_Body.L5_Mental.Memory.aging_clock import BiologicalClock
from Core.1_Body.L5_Mental.Memory.qualia_layer import QualiaTag

class DecayManager:
    """
    Manages the entropy of the Phenomenal Layer (Qualia).
    The Akashic Layer (Facts) is immune to this manager.
    """

    def __init__(self, clock: BiologicalClock):
        self.clock = clock

    def calculate_survival_probability(self, last_access_timestamp: float, stability: float) -> float:
        """
        Calculates the probability (0.0~1.0) that a memory survives today.

        Args:
            last_access_timestamp: Unix time when the Qualia was last touched.
            stability: Current vividness/importance of the Qualia (0.0~1.0).

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
        # S = Stability (Strength of memory)

        # We adjust S based on the Qualia's inherent importance AND biological entropy
        # High Entropy Rate (Childhood) = Lower effective stability (Faster forgetting)
        effective_stability = (stability * 5.0) * (1.0 - (entropy_base_rate * 0.5))

        # Protect against div by zero
        if effective_stability < 0.01:
            effective_stability = 0.01

        retention = math.exp(-elapsed_years / effective_stability)

        return retention

    def apply_entropy_to_qualia(self, qualia: QualiaTag) -> Tuple[str, float]:
        """
        Applies decay logic to a subjective experience (QualiaTag).

        Returns:
            (Status, NewVividness)
            Status: 'KEPT', 'FADED', 'FORGOTTEN'
        """
        survival_prob = self.calculate_survival_probability(qualia.last_recalled_at, qualia.vividness)

        # Chaos Factor (Random variability)
        roll = random.random()

        if roll > survival_prob:
            # Decay Event
            if qualia.vividness > 0.2:
                # Fade: It becomes fuzzy
                # e.g. "I know I was happy, but I can't feel the intensity anymore."
                decay_amount = 0.1 * (1.0 + random.random()) # Random erosion
                qualia.decay(decay_amount)
                return "FADED", qualia.vividness
            else:
                # Death: It vanishes
                qualia.vividness = 0.0
                return "FORGOTTEN", 0.0

        return "KEPT", qualia.vividness
