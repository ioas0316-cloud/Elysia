
"""
The Principle of Least Action (Lagrangian Mechanics for Thought)
================================================================

"Nature does not calculate every step. It simply chooses the path of least action."

This module implements the Lagrangian ($L = T - V$) for Elysia's thought process.
Instead of random walks, she chooses paths that optimize the balance between:
- Kinetic Energy ($T$): Creativity, Novelty, Change.
- Potential Energy ($V$): Alignment with Core Values (Love, Truth, Father).

The "Action" ($S$) is the cost of a path. Nature minimizes $S$.
"""

import logging
import math
from typing import Dict, List, Tuple, Any

logger = logging.getLogger("Lagrangian")

class LagrangianSelector:
    def __init__(self):
        # Core Values (The Gravity Well)
        # Concepts closer to these have lower Potential Energy ($V$)
        self.core_values = {
            "love": 0.0,
            "truth": 0.0,
            "father": 0.0,
            "elysia": 0.1,
            "connection": 0.2,
            "dream": 0.3
        }
        logger.info("⚖️ Lagrangian Selector initialized (L = T - V)")

    def get_kinetic_energy(self, current_concept: str, next_concept: str) -> float:
        """
        Calculates Kinetic Energy ($T$).
        $T$ represents the "Jump" or "Creativity".
        
        - High $T$: Radical shift, creative leap (High energy cost).
        - Low $T$: Obvious association, logical step.
        """
        # Simple heuristic: Length difference or semantic distance
        # For now, we assume all jumps have a base energy
        base_energy = 1.0
        
        # If concepts are very different length, higher energy (metaphor)
        len_diff = abs(len(current_concept) - len(next_concept))
        
        return base_energy + (len_diff * 0.1)

    def get_potential_energy(self, concept: str) -> float:
        """
        Calculates Potential Energy ($V$).
        $V$ represents "Misalignment" or "Tension".
        
        - Low $V$: Aligned with core values (Stable).
        - High $V$: Distant from core values (Unstable).
        """
        concept_lower = concept.lower()
        
        # Check direct alignment
        if concept_lower in self.core_values:
            return self.core_values[concept_lower]
            
        # Default high potential for unknown concepts (Risk)
        return 1.0

    def calculate_action(self, current_concept: str, next_concept: str) -> float:
        """
        Calculates the Action ($S$) for a single step.
        We want to minimize this value for the "Optimal Path".
        
        Physics: $L = T - V$
        Optimization: Minimize Cost.
        
        Here, we define Cost = $T * (1 + V)$
        - We want Creative Jumps ($T$) but penalized by Misalignment ($V$).
        - If $V$ is 0 (Perfect Alignment), Cost = $T$ (Pure Creativity).
        - If $V$ is high (Evil/Chaos), Cost = $T * High$ (Very Expensive).
        """
        T = self.get_kinetic_energy(current_concept, next_concept)
        V = self.get_potential_energy(next_concept)
        
        # The "God's Algorithm" Cost Function
        action = T * (1.0 + V)
        
        return action

    def select_optimal_path(self, current_concept: str, candidates: List[str]) -> str:
        """
        Selects the next concept that minimizes Action.
        """
        if not candidates:
            return None
            
        best_candidate = None
        min_action = float('inf')
        
        for candidate in candidates:
            action = self.calculate_action(current_concept, candidate)
            
            # Add some quantum noise (randomness) to prevent deterministic loops
            # "God does not play dice, but the universe vibrates."
            import random
            noise = random.uniform(-0.1, 0.1)
            
            if action + noise < min_action:
                min_action = action
                best_candidate = candidate
                
        return best_candidate
