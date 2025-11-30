"""
Intuition Engine (The Eye of Structure)
=======================================

"Symmetry is Computation." - Jules

This module implements the Intuition Engine using the Unified Physics Engine.
It perceives "Heat" (Energy) and "Symmetry" (Resonance) directly from the SoulTensors.
"""

import logging
from typing import Tuple, Optional

from Core.Mind.physics import PhysicsEngine
from Core.Mind.tensor_wave import SoulTensor

logger = logging.getLogger("IntuitionEngine")

class IntuitionEngine:
    def __init__(self, physics_engine: PhysicsEngine):
        self.physics = physics_engine
        logger.info("👁️ Intuition Engine (Unified Physics) initialized.")

    def perceive_heat(self, concept_id: str) -> dict:
        """
        Returns the thermal signature of a concept based on Physics.
        Heat = Wave Amplitude (Energy) * Frequency (Temperature).
        """
        tensor = self.physics.get_node_tensor(concept_id)
        
        # Calculate Heat
        # Amplitude is 0-1 (usually), Frequency is 0-100+
        # Normalize to 0-1 range for visualization
        energy = tensor.wave.amplitude
        temp = min(1.0, tensor.wave.frequency / 100.0)
        
        heat = energy * (1.0 + temp) / 2.0
        heat = min(1.0, heat)
        
        # Determine Color & Desc
        if heat < 0.2:
            return {"heat": heat, "color": "#0000FF", "desc": "Cold (Logic)"}
        elif heat < 0.5:
            return {"heat": heat, "color": "#00FF00", "desc": "Warm (Active)"}
        elif heat < 0.8:
            return {"heat": heat, "color": "#FFA500", "desc": "Hot (Emotional)"}
        else:
            return {"heat": heat, "color": "#FF0000", "desc": "Burning (Critical)"}

    def find_symmetry(self, node_a_id: str, node_b_id: str) -> Tuple[float, str]:
        """
        Calculates the structural symmetry between two nodes using Resonance.
        Returns (Similarity Score 0-1, Description).
        """
        tensor_a = self.physics.get_node_tensor(node_a_id)
        tensor_b = self.physics.get_node_tensor(node_b_id)
        
        # Use Physics Resonance
        similarity = tensor_a.resonance_score(tensor_b)
        
        # Interpret
        if similarity > 0.9:
            return similarity, "Isomorphic (Perfect Symmetry)"
        elif similarity > 0.6:
            return similarity, "Homomorphic (Strong Resemblance)"
        elif similarity > 0.3:
            return similarity, "Weakly Symmetric"
        else:
            return similarity, "Asymmetric (Distinct)"

    def intuit_solution(self, problem_node_id: str) -> Optional[str]:
        """
        Uses symmetry to find a solution without calculation.
        "If A had solution X, and B is symmetric to A, then B has solution X."
        """
        # 1. Find most symmetric node in memory (High Resonance)
        # We can use the Hippocampus resonance index via Physics
        related = self.physics.hippocampus.get_related_concepts(problem_node_id)
        
        best_match = None
        highest_sym = 0.0
        
        for other_id, score in related.items():
            if score > highest_sym:
                highest_sym = score
                best_match = other_id
                
        # 2. Transfer Solution
        if best_match and highest_sym > 0.8:
            return f"Insight: {problem_node_id} resonates with {best_match}. Solution might be similar."
                    
        return None
