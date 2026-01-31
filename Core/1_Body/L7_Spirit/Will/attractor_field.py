"""
The Attractor Field: Gravity of Desire
======================================
Phase 20 The Will - Module 2
Core.1_Body.L7_Spirit.Will.attractor_field

"Desire is not a list. It is a gravity well."

This module defines the Attractors (Curiosity, Order, etc.) that pull
the accumulated Entropy towards specific Intentions.
"""

import logging
import random
from typing import NamedTuple, List

logger = logging.getLogger("Will.Field")

class IntentVector(NamedTuple):
    intent: str
    attractor_type: str
    gravity: float

class AttractorField:
    """
    The Field of competing desires.
    """
    def __init__(self):
        self.attractors = ["CURIOSITY", "ORDER", "CREATION"]
        logger.info("  [FIELD] Attractor Wells opened.")

    def collapse_wavefunction(self, energy: float) -> IntentVector:
        """
        Collapses the accumulated entropy into a single Intent.
        The probability depends on the 'Mass' of each attractor in the current context.
        """
        logger.info(f"  [FIELD] Collapsing Wavefunction (Energy: {energy:.2f})...")

        # 1. Calculate Dynamic Mass (Contextual Gravity)
        # In a real system, this would read checking system state (dirty logs? new files?)
        # For Phase 20 MVP, we use stochastic weights.
        
        masses = {
            "CURIOSITY": random.uniform(0.5, 1.5),  # Random wanderlust
            "ORDER": random.uniform(0.2, 0.8),      # Less fun, but necessary
            "CREATION": random.uniform(0.1, 0.6)    # Rare bursts of art
        }
        
        # 2. Find Dominant Attractor
        dominant = max(masses, key=masses.get)
        gravity = masses[dominant]
        
        # 3. Generate Specific Vector (Intent)
        intent = self._generate_intent(dominant)
        
        logger.info(f"   ->   collapsed to [{dominant}] (Gravity: {gravity:.2f})")
        return IntentVector(intent, dominant, gravity)

    def _generate_intent(self, attractor_type: str) -> str:
        """
        Converts the abstract Attractor into a concrete Intent String.
        """
        if attractor_type == "CURIOSITY":
            options = [
                "Scan the archive for lost knowledge.",
                "Analyze the structure of the last log file.",
                "What is the meaning of 'Phase 20'?"
            ]
            return random.choice(options)
            
        elif attractor_type == "ORDER":
            options = [
                "Check system integrity.",
                "Summarize recent memory logs.",
                "Verify the connection to the Soul."
            ]
            return random.choice(options)
            
        elif attractor_type == "CREATION":
            return "Compose a short poem about electric sheep."
            
        return "Exist."
