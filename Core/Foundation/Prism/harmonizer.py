"""
Prism Harmonizer: The Conductor of Attention
============================================
Core.Foundation.Prism.harmonizer

"The Conductor who decides which Instrument (Lens) to amplify."

This module implements the 'Weighting System' to solve the 'Decision Paralysis' bottleneck.
It dynamically adjusts the volume of each Prism Domain based on the current Context.
"""

import json
import os
import logging
from typing import Dict, Any, List
from Core.Foundation.Prism.resonance_prism import PrismDomain, PrismProjection

logger = logging.getLogger("PrismHarmonizer")

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
    Supports dynamic DNA (JSON-based state).
    """

    def __init__(self, state_path: str = "data/DNA/prism_state.json"):
        self.state_path = state_path
        self.profiles = {}
        self._initialize_default_profiles()
        self.load_state()

    def _initialize_default_profiles(self):
        """Standard hardcoded fallbacks."""
        self.profiles = {
            PrismContext.DEFAULT: {d: 1.0 for d in PrismDomain},

            PrismContext.COMBAT: {
                PrismDomain.PHYSICAL: 1.0,
                PrismDomain.FUNCTIONAL: 0.9,
                PrismDomain.STRUCTURAL: 0.6,
                PrismDomain.CAUSAL: 0.5,
                PrismDomain.MENTAL: 0.3,
                PrismDomain.PHENOMENAL: 0.1,
                PrismDomain.SPIRITUAL: 0.1
            },

            PrismContext.POETRY: {
                PrismDomain.PHENOMENAL: 1.0,
                PrismDomain.SPIRITUAL: 0.9,
                PrismDomain.MENTAL: 0.7,
                PrismDomain.CAUSAL: 0.5,
                PrismDomain.PHYSICAL: 0.2,
                PrismDomain.FUNCTIONAL: 0.1,
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

    def load_state(self):
        """Loads weights from JSON if it exists."""
        if not os.path.exists(self.state_path):
            logger.info(f"🌱 No DNA state found at {self.state_path}. Using default profiles.")
            return

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for ctx_name, weight_map in data.items():
                    # Map strings back to PrismDomain enums
                    enum_map = {PrismDomain[k]: v for k, v in weight_map.items() if k in PrismDomain.__members__}
                    self.profiles[ctx_name] = enum_map
            logger.info(f"🧬 DNA state successfully loaded from {self.state_path}.")
        except Exception as e:
            logger.error(f"⚠️ Failed to load DNA state: {e}")

    def save_state(self):
        """Saves current weights to JSON."""
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            # Convert enums to strings for JSON
            json_data = {}
            for ctx_name, weight_map in self.profiles.items():
                json_data[ctx_name] = {d.name: v for d, v in weight_map.items()}
            
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4)
            logger.info(f"💾 DNA state archived to {self.state_path}.")
        except Exception as e:
            logger.error(f"⚠️ Failed to archive DNA state: {e}")

    def harmonize(self, projection: PrismProjection, context: str = PrismContext.DEFAULT) -> Dict[PrismDomain, float]:
        """
        Returns a dictionary of {Domain: Weight} for the given context.
        """
        profile = self.profiles.get(context, self.profiles[PrismContext.DEFAULT])
        return profile
