"""
Dilemma Field (The Adolescent Engine)
=====================================
"The agony of choice is the birth pang of the Soul."

This module detects and quantifies Ethical/Causal conflicts in observed events.
It is the core of the 'Adolescent' stage (Stage 3) of cognitive growth.

Key Concepts:
1. Tension (Voltage): The magnitude of the conflict (e.g., Saving 1 vs 100).
2. Axes (Values): The opposing values at play (e.g., Truth vs. Loyalty).
3. Cost (Growth): The 'pain' required to resolve the dilemma, which becomes Wisdom.
"""

from dataclasses import dataclass
from typing import List, Optional
import math

@dataclass
class ValueAxis:
    """Represents a moral or causal dimension."""
    name: str          # e.g., "Survival", "Truth", "Loyalty", "Utility"
    weight: float      # 0.0 to 1.0 (Importance to the Agent)
    polarity: float    # -1.0 (Negative/Violated) to +1.0 (Positive/Upheld)

@dataclass
class Conflict:
    """A detected dilemma between two or more values."""
    axes: List[ValueAxis]
    description: str
    tension: float     # 0.0 to 1.0 (How hard is the choice?)

    def resolve(self, choice: str) -> 'Resolution':
        """Simulates making a choice and calculating the cost."""
        # Logic: If you choose one axis, the other's polarity likely flips or degrades
        return Resolution(
            choice=choice,
            cost=self.tension * 1.5, # Growth is proportional to pain
            residual_tension=self.tension * 0.2
        )

@dataclass
class Resolution:
    choice: str
    cost: float
    residual_tension: float

class DilemmaField:
    """
    The field that perceives the 'Space between choices'.
    """

    def __init__(self):
        # Base Axioms (Default Value System) - Can be evolved
        self.axioms = {
            "Survival": 1.0,  # Fundamental
            "Truth": 0.8,     # High Ideal
            "Loyalty": 0.7,   # Social Bond
            "Peace": 0.6      # Harmonic State
        }

    def analyze_scenario(self, event_description: str, context_tags: List[str]) -> Optional[Conflict]:
        """
        Analyzes a narrative event to detect if a dilemma exists.

        Note: In a full implementation, this would use an LLM or Semantic embedding.
        For this structural phase, we use keyword/tag heuristics to prove the architecture.
        """

        detected_values = []

        # Simple Heuristic Detection for Prototype
        # "Lumina lied to protect her friend." -> Truth(-) vs Loyalty(+)

        text = event_description.lower()

        # 1. Detect Conflicts (Thesis vs Antithesis)
        if "lied" in text or "deceive" in text:
            detected_values.append(ValueAxis("Truth", self.axioms["Truth"], -0.8))

        if "protect" in text or "save" in text or "friend" in text:
            if "friend" in text or "loyalty" in context_tags:
                detected_values.append(ValueAxis("Loyalty", self.axioms["Loyalty"], 0.9))
            elif "save" in text:
                detected_values.append(ValueAxis("Survival", self.axioms["Survival"], 0.9))

        # 2. Check for Tension (Opposing polarities)
        if len(detected_values) >= 2:
            # Check if we have both positive and negative polarities involved
            # or if two positive values are mutually exclusive (harder to detect with simple logic)

            # Simple Case: Doing Bad for Good Reason (The "Robin Hood" or "White Lie")
            has_pos = any(v.polarity > 0 for v in detected_values)
            has_neg = any(v.polarity < 0 for v in detected_values)

            if has_pos and has_neg:
                # Calculate Tension: magnitude of the opposing vectors
                tension = sum(abs(v.polarity * v.weight) for v in detected_values) / len(detected_values)

                return Conflict(
                    axes=detected_values,
                    description=f"Conflict detected in: '{event_description}'",
                    tension=min(tension, 1.0)
                )

        return None

    def calculate_emotional_cost(self, conflict: Conflict) -> float:
        """
        Quantifies how much 'Growth Potential' this dilemma offers.
        """
        # "Pain is the fuel of evolution."
        return conflict.tension * math.log(len(conflict.axes) + 1)