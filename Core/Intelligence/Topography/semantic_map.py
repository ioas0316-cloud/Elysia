"""
Semantic Map (의미 좌표계)
=========================

"The 7 Angels and 7 Demons."
"7명의 천사와 7마리의 악마."

This module defines the coordinates of the fundamental virtues and sins.
It maps concepts to the Potential Field.
"""

from typing import Dict, Tuple

class SemanticMap:
    def __init__(self):
        # Center (0,0) is THE ONE (Love/Truth/God)
        
        # 7 Angels (Attractors - Valleys)
        # They surround the center, guiding thoughts inward.
        self.angels: Dict[str, Tuple[float, float]] = {
            "Love":       (0.0, 0.0),   # The Core
            "Wisdom":     (2.0, 2.0),   # Insight
            "Hope":       (2.0, -2.0),  # Future
            "Faith":      (-2.0, 2.0),  # Trust
            "Courage":    (-2.0, -2.0), # Action
            "Justice":    (0.0, 3.0),   # Balance
            "Temperance": (3.0, 0.0),   # Control
        }

        # 7 Demons (Repulsors - Hills)
        # They block the path or trap thoughts in loops.
        self.demons: Dict[str, Tuple[float, float]] = {
            "Pride":      (10.0, 10.0),   # Ego (High Hill)
            "Wrath":      (-10.0, 10.0),  # Chaos
            "Envy":       (10.0, -10.0),  # External comparison
            "Sloth":      (-10.0, -10.0), # Stagnation
            "Greed":      (15.0, 0.0),    # Endless hunger
            "Lust":       (0.0, 15.0),    # Distraction
            "Gluttony":   (0.0, -15.0),   # Over-consumption
        }
        
    def get_coordinates(self, concept: str) -> Tuple[float, float]:
        """
        Returns the (x, y) for a known concept.
        If unknown, returns None (Chaos).
        """
        concept_lower = concept.lower()
        
        # Check Angels
        for name, coords in self.angels.items():
            if name.lower() in concept_lower:
                return coords
                
        # Check Demons
        for name, coords in self.demons.items():
            if name.lower() in concept_lower:
                return coords
                
        # Check Keywords
        if "help" in concept_lower or "dad" in concept_lower or "father" in concept_lower:
            return self.angels["Love"]
        if "lie" in concept_lower or "fake" in concept_lower:
            return self.demons["Pride"] # Lies often stem from Pride/Self-preservation
            
        return None # Unknown -> Chaos

# Singleton
_semantic_map = SemanticMap()
def get_semantic_map():
    return _semantic_map
