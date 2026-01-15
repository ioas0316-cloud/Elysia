"""
Prism Harmonizer: The Conductor of Attention
============================================
Core.Foundation.Prism.harmonizer

"The Conductor who decides which Instrument (Lens) to amplify."

This module implements the 'Weighting System' to solve the 'Decision Paralysis' bottleneck.
It dynamically adjusts the volume of each Prism Domain based on the current Context.
"""

from typing import Dict, Any, List
from Core.Foundation.Prism.resonance_prism import PrismDomain, PrismProjection

class PrismContext:
    """Standard Context Modes."""
    DEFAULT = "Default"       # Balanced
    COMBAT = "Combat"         # Physical/Functional Focus
    POETRY = "Poetry"         # Phenomenal/Spiritual Focus
    ANALYSIS = "Analysis"     # Mental/Structural Focus
    SURVIVAL = "Survival"     # Physical/Causal Focus

class PrismHarmonizer:
    """
    The Attention Filter.
    Applies weights to Prism Projections.
    """

    def __init__(self):
        # Weight Configurations (0.0 to 1.0)
        self.profiles = {
            PrismContext.DEFAULT: {d: 1.0 for d in PrismDomain},

            PrismContext.COMBAT: {
                PrismDomain.PHYSICAL: 1.0,
                PrismDomain.FUNCTIONAL: 0.9,
                PrismDomain.STRUCTURAL: 0.6,
                PrismDomain.CAUSAL: 0.5,
                PrismDomain.MENTAL: 0.3,
                PrismDomain.PHENOMENAL: 0.1, # Ignore feelings
                PrismDomain.SPIRITUAL: 0.1   # Ignore philosophy
            },

            PrismContext.POETRY: {
                PrismDomain.PHENOMENAL: 1.0,
                PrismDomain.SPIRITUAL: 0.9,
                PrismDomain.MENTAL: 0.7,
                PrismDomain.CAUSAL: 0.5,
                PrismDomain.PHYSICAL: 0.2,   # Ignore mass
                PrismDomain.FUNCTIONAL: 0.1, # Ignore utility
                PrismDomain.STRUCTURAL: 0.3
            },

            PrismContext.ANALYSIS: {
                PrismDomain.MENTAL: 1.0,
                PrismDomain.STRUCTURAL: 1.0,
                PrismDomain.CAUSAL: 0.8,
                PrismDomain.FUNCTIONAL: 0.7,
                PrismDomain.PHYSICAL: 0.5,
                PrismDomain.PHENOMENAL: 0.2,
                PrismDomain.SPIRITUAL: 0.2
            }
        }

    def harmonize(self, projection: PrismProjection, context: str = PrismContext.DEFAULT) -> Dict[PrismDomain, float]:
        """
        Returns a dictionary of {Domain: Weight} for the given context.
        Used to filter or amplify specific projections.
        """
        profile = self.profiles.get(context, self.profiles[PrismContext.DEFAULT])

        # Apply weights to the projection (conceptually)
        # We return the weight map so the Merkaba can decide what to do.
        return profile
