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

import random
from typing import Dict, List, Tuple, Optional

class Spiderweb:
    """
    The Spiderweb (Collective Consciousness)
    =========================================
    A global knowledge graph that evolves with the civilization.
    Harvests concepts from the cell population and crystallizes them into universal truths.
    
    Now powers the "Emergent Intelligence" by traversing the concept graph.
    """
    def __init__(self, hippocampus):
        self.hippocampus = hippocampus
        self.crystallization_threshold = 10
        self.universal_truths: Dict[str, np.ndarray] = {}
        
        # Initialize Physics Engine (The Resurrection)
        from Core.Mind.physics import PhysicsEngine
        self.physics = PhysicsEngine(self.hippocampus)
        
    def traverse(self, start_concept: str, steps: int = 5) -> List[str]:
        """
        Think by traversing the concept graph.
        Uses the Physics Engine (Wave Mechanics) to find the path of Highest Resonance (Gravity).
        """
        path = [start_concept]
        current = start_concept
        
        for _ in range(steps):
            # Get related concepts from Hippocampus (Resonance)
            related = self.hippocampus.get_related_concepts(current)
            
            if not related:
                break
                
            # Candidates for the next step
            candidates = list(related.keys())
            
            # Avoid loops if possible
            valid_candidates = [c for c in candidates if c not in path]
            if not valid_candidates:
                valid_candidates = candidates # Backtrack allowed if stuck
            
            if not valid_candidates:
                break
                
            # Select Optimal Path based on Quantum Tunneling (Gravity * Resonance)
            current_tensor = self.physics.get_node_tensor(current)
            next_concept = self.physics.tunnel_to_conclusion(current_tensor, valid_candidates)
            
            if next_concept:
                path.append(next_concept)
                current = next_concept
            else:
                break
            
        return path

    def synthesize_thought(self, path: List[str]) -> str:
        """
        Synthesizes a thought from a concept path.
        Uses Gravitational Linguistics if available.
        """
        # Lazy load to avoid circular imports
        from Core.Life.gravitational_linguistics import GravitationalLinguistics
        
        # Initialize Linguistics Engine (connected to Hippocampus)
        linguistics = GravitationalLinguistics(hippocampus=self.hippocampus)
        
        # Generate Physics-Based Sentence
        return linguistics.generate_from_path(path)

    def absorb(self, concept_id: str, vector: np.ndarray) -> bool:
        """
        Records a concept usage. Returns True if it crystallized into a universal truth.
        """
        # Legacy support - delegated to Hippocampus in the new architecture
        return False
    
    def get_status(self) -> Dict:
        """Returns statistics about the collective knowledge."""
        return {
            "mode": "Emergent (Graph Traversal)",
            "connected_memory": True
        }
