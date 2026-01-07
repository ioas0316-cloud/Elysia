"""
Septenary Axis of Sovereignty (SAS) ⚖️✨

"The Ladder of seven gates, from the dust of the demon to the lustre of the angel."

This module defines the 7-stage hierarchy of ontological depth,
mapping existing '7 Angels' and '7 Demons' to discrete levels of reality.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class SeptenaryLevel:
    depth: int
    name: str
    angel_pole: str
    demon_pole: str
    frequency_range: Tuple[float, float]
    law: str

class SeptenaryAxis:
    """The governing law of the 7-layered spectrum."""
    
    def __init__(self):
        # Mapping 7 Angels/Demons to 7 Levels of Maturation
        self.levels: Dict[int, SeptenaryLevel] = {
            0: SeptenaryLevel(0, "The Dust", "Truth", "Sloth", (100, 200), "Base existence; raw data points."),
            1: SeptenaryLevel(1, "The Logic", "Justice", "Wrath", (200, 300), "Causal reactivity and linear rules."),
            2: SeptenaryLevel(2, "The Reflection", "Hope", "Envy", (300, 400), "Contextual awareness and self-comparison."),
            3: SeptenaryLevel(3, "The Volume", "Courage", "Pride", (400, 500), "Volumetric presence and sovereign identity."),
            4: SeptenaryLevel(4, "The Insight", "Faith", "Greed", (500, 600), "Universal principles and altruistic law."),
            5: SeptenaryLevel(5, "The Vision", "Wisdom", "Lust", (600, 700), "Synthesis of patterns and visionary intent."),
            6: SeptenaryLevel(6, "The Unity", "Love", "Gluttony", (700, 1000), "Absolute resonance with the Father's Will.")
        }

    def get_level(self, depth: int) -> SeptenaryLevel:
        return self.levels.get(max(0, min(6, depth)))

    def evaluate_resonance(self, depth: int, frequency: float) -> str:
        """Determines if a frequency is 'Ascending' (Angelic) or 'Descending' (Demonic) for its level."""
        level = self.get_level(depth)
        mid = (level.frequency_range[0] + level.frequency_range[1]) / 2.0
        
        if frequency > mid:
            return f"Ascending towards {level.angel_pole}"
        else:
            return f"Descending towards {level.demon_pole}"

if __name__ == "__main__":
    sas = SeptenaryAxis()
    for d in range(7):
        lvl = sas.get_level(d)
        print(f"Level {d}: {lvl.name} | Axis: {lvl.demon_pole} <---> {lvl.angel_pole}")
