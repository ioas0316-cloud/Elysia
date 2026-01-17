"""
The Quality Gate (Threshold Module)
===================================
Core.Foundation.Prism.quality_gate

"Filtering the Fog."

This module implements the check for 'Coherence'.
"""

from .integrating_lens import Insight

class QualityGate:
    """
    The Threshold Guardian.
    """

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def check_insight(self, insight: Insight) -> bool:
        """
        Returns True if the thought is clear (Coherent).
        Returns False if it is foggy (Needs Rumination).
        """
        return insight.coherence >= self.threshold
