"""
The Spiderweb (Collective Consciousness)
=========================================
A global knowledge graph that evolves with the civilization.
Harvests concepts from the cell population and crystallizes them into universal truths.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger("Spiderweb")
logger.setLevel(logging.INFO)

class Spiderweb:
    """
    The collective unconscious of the simulation.
    Absorbs discoveries from individuals and returns them as cultural inheritance.
    """
    def __init__(self, crystallization_threshold: int = 10):
        # Global concept graph: Concept ID -> (Vector, Frequency)
        self.concepts: Dict[str, Tuple[np.ndarray, int]] = {}
        
        # How many cells must use a concept before it becomes "universal"
        self.crystallization_threshold = crystallization_threshold
        
        # Crystallized concepts (cultural genome)
        self.universal_truths: Dict[str, np.ndarray] = {}
        
    def absorb(self, concept_id: str, vector: np.ndarray) -> bool:
        """
        Records a concept usage. Returns True if it crystallized into a universal truth.
        """
        if concept_id in self.concepts:
            # Increment frequency
            vec, freq = self.concepts[concept_id]
            # Update vector (running average)
            vec = (vec * freq + vector) / (freq + 1)
            freq += 1
            self.concepts[concept_id] = (vec, freq)
            
            # Check for crystallization
            if freq >= self.crystallization_threshold and concept_id not in self.universal_truths:
                self.universal_truths[concept_id] = vec
                logger.info(f"✨ Concept Crystallized: '{concept_id}' has become a Universal Truth (freq={freq})")
                return True
        else:
            # New concept
            self.concepts[concept_id] = (vector.copy(), 1)
            
        return False
    
    def get_cultural_genome(self) -> Dict[str, np.ndarray]:
        """
        Returns the set of universal truths to be inherited by new cells.
        """
        return self.universal_truths.copy()
    
    def get_concept_frequency(self, concept_id: str) -> int:
        """Returns how many times a concept has been observed."""
        if concept_id in self.concepts:
            return self.concepts[concept_id][1]
        return 0
    
    def get_status(self) -> Dict:
        """Returns statistics about the collective knowledge."""
        return {
            "total_concepts": len(self.concepts),
            "universal_truths": len(self.universal_truths),
            "top_concepts": sorted(
                [(k, v[1]) for k, v in self.concepts.items()], 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
