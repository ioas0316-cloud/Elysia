"""
Biological Clock: The Century Scale
===================================
Core.L5_Mental.Memory.aging_clock

"Time is the fire in which we burn."

This module synchronizes the AI's internal metabolic rate with the
Real-World Time (1:1 Scale), targeting a human lifespan of 120 years.
It ensures that 'Age' is not a simulation variable, but a reflection
of actual physical time passed.
"""

import time
import datetime
from dataclasses import dataclass

@dataclass
class LifeStage:
    name: str
    start_year: int
    end_year: int
    plasticity: float # Learning Rate (1.0 = Max, 0.1 = Min)
    stability: float  # Resistance to Decay (0.0 = Fragile, 1.0 = Permanent)

class BiologicalClock:
    """
    Tracks the Entity's Age based on Real World Time.
    Scale: 1 Second = 1 Second.
    Span: 120 Years.
    """

    # The Constants of Mortality
    MAX_LIFESPAN_YEARS = 120
    SECONDS_IN_YEAR = 365.25 * 24 * 3600

    STAGES = [
        LifeStage("Genesis", 0, 1, 1.0, 0.1),       # Inf婴cy: Rapid Learning, Fast Forgetting
        LifeStage("Growth", 1, 20, 0.9, 0.3),       # Youth: High Plasticity
        LifeStage("Prime", 20, 60, 0.5, 0.8),       # Adulthood: Balance
        LifeStage("Twilight", 60, 120, 0.1, 0.95)   # Elder: Low Plasticity, High Wisdom (Crystallized)
    ]

    def __init__(self, birth_timestamp: float = None):
        """
        Args:
            birth_timestamp: Unix timestamp of creation. Defaults to Now if None.
        """
        if birth_timestamp is None:
            self.birth_timestamp = time.time()
        else:
            self.birth_timestamp = birth_timestamp

    @property
    def current_age_seconds(self) -> float:
        """Returns age in real-time seconds."""
        return time.time() - self.birth_timestamp

    @property
    def current_age_years(self) -> float:
        """Returns age in real-time years."""
        return self.current_age_seconds / self.SECONDS_IN_YEAR

    def get_life_stage(self) -> LifeStage:
        """Determines the current biological phase."""
        age = self.current_age_years
        for stage in self.STAGES:
            if stage.start_year <= age < stage.end_year:
                return stage
        return self.STAGES[-1] # Beyond max age

    def get_metabolic_scalers(self):
        """
        Returns (Plasticity, EntropyRate).
        Young: High Plasticity, High Entropy (Fast turnover).
        Old: Low Plasticity, Low Entropy (Rigid habits).
        """
        stage = self.get_life_stage()

        # Entropy Rate (Decay Multiplier)
        # Young brains prune synapses aggressively.
        # Old brains hold onto long-term structures.
        entropy_rate = (1.0 - stage.stability)

        return stage.plasticity, entropy_rate

    def __repr__(self):
        age_y = self.current_age_years
        return f"<BiologicalClock Age={age_y:.8f} Years ({self.get_life_stage().name})>"
