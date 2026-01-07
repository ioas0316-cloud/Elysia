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
    domain: str  # Body (1-3), Soul (4-6), Spirit (7)
    name: str
    angel_pole: str
    demon_pole: str
    frequency_range: Tuple[float, float]
    law: str

class SeptenaryAxis:
    """The governing law of the 9-layered Fractal Trinity (Body -> Soul -> Spirit)."""
    
    def __init__(self):
        # 1-9 Hierarchy based on the 3x3 Fractal Trinity
        self.levels: Dict[int, SeptenaryLevel] = {
            # --- DOMAIN: BODY (Physical Mastery) ---
            1: SeptenaryLevel(1, "Body", "Technique", "Truth", "Sloth", (100, 200), "Physical foundation; apprenticeship."),
            2: SeptenaryLevel(2, "Body", "Reason", "Justice", "Wrath", (200, 300), "Understanding physical causality."),
            3: SeptenaryLevel(3, "Body", "Meaning", "Hope", "Envy", (300, 400), "Physical labor as a ritual of value."),
            
            # --- DOMAIN: SOUL (Mental Sovereignty) ---
            4: SeptenaryLevel(4, "Soul", "Technique", "Courage", "Pride", (400, 500), "Mental willpower; expert craftsmanship."),
            5: SeptenaryLevel(5, "Soul", "Reason", "Faith", "Greed", (500, 600), "Structuring logical laws of the mind."),
            6: SeptenaryLevel(6, "Soul", "Meaning", "Wisdom", "Lust", (600, 700), "Visionary purpose; identifying as a Pro."),
            
            # --- DOMAIN: SPIRIT (Divine Resonance) ---
            7: SeptenaryLevel(7, "Spirit", "Technique", "Temperance", "Gluttony", (700, 800), "Spiritual discipline; beginning of mastery."),
            8: SeptenaryLevel(8, "Spirit", "Reason", "Fortitude", "Avarice", (800, 900), "Causal alignment with the Logos."),
            9: SeptenaryLevel(9, "Spirit", "Meaning", "Love", "Despair", (900, 1000), "Absolute unity and eternal value.")
        }

    def get_level(self, depth: int) -> SeptenaryLevel:
        return self.levels.get(max(1, min(9, depth)))

    def get_rank(self, depth: int) -> str:
        """Translates depth into fractal ranks."""
        if depth <= 3: return "Novice/Apprentice"
        if depth <= 6: return "Expert/Professional"
        return "Master/Divine"

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
