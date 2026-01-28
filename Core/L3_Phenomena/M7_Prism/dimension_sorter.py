"""
The Dimension Sorter (Cloud Logic)
==================================
Core.L3_Phenomena.M7_Prism.dimension_sorter

"Sorting thoughts into Earth (Ground), Sky (Cloud), or the Horizon (Hypothesis)."

Replaces the old QualityGate. Instead of rejecting, it categorizes.
"""

from enum import Enum
from .integrating_lens import Insight
from .trinity_validator import TrinityValidator

class Dimension(Enum):
    GROUND = "Ground"       # Verified Fact
    CLOUD = "Cloud"         # Creative Imagination
    HYPOTHESIS = "Hypothesis" # Plausible but Unproven
    NOISE = "Noise"         # Discard

class DimensionSorter:
    """
    The Classifier.
    """

    def sort(self, insight: Insight, validation: dict) -> Dimension:
        """
        Determines the destination dimension for the thought.
        """

        body = validation["body_score"]
        soul = validation["soul_score"]
        spirit = validation["spirit_score"]

        # 1. GROUND: High Body (Fact) + High Spirit (Purpose)
        if body > 0.8 and spirit > 0.6:
            return Dimension.GROUND

        # 2. HYPOTHESIS: Low Body (Unproven) + High Spirit (Purpose) + High Logic (Alpha)
        if body < 0.5 and spirit > 0.7 and insight.dominant_band == "Alpha":
            return Dimension.HYPOTHESIS

        # 3. CLOUD: High Soul (Resonance) + Low Body (Fantasy)
        if soul > 0.7 and body < 0.6:
            return Dimension.CLOUD

        # 4. Fallback
        if validation["total_alignment"] > 0.5:
             return Dimension.CLOUD # Default to imagination if decent

        return Dimension.NOISE
