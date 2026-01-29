import jax.numpy as jnp
from typing import List, Dict, Tuple
from Core.L5_Cognition.Reasoning.logos_bridge import LogosBridge

class InferentialManifold:
    """
    [L5_COGNITION: TOPOLOGICAL_INFERENCE]
    Reasoning as a spatial navigation of Phase Gradients.
    
    Transforms the "Is it this or that?" question into a vector-field search.
    """
    def __init__(self):
        self.bridge = LogosBridge()
        
    def explore_possibilities(self, current_field: jnp.ndarray, candidates: List[str]) -> Dict[str, float]:
        """
        [THE EXPLORATION PULSE]
        Calculates the resonance (Phase Match) for each candidate.
        Discrimination = The Magnitude of the Gradient.
        """
        results = {}
        norm_c = jnp.linalg.norm(current_field) + 1e-6
        
        for name in candidates:
            target_vec = self.bridge.recall_concept_vector(name)
            norm_t = jnp.linalg.norm(target_vec) + 1e-6
            
            # Resonance as Cosine Similarity (Phase Alignment)
            resonance = jnp.dot(current_field, target_vec) / (norm_c * norm_t)
            
            # We treat the resonance as the 'Probability of Truth' in the manifold
            results[name] = float(resonance)
            
        return results

    def resolve_inference(self, current_field: jnp.ndarray, candidates: List[str]) -> Tuple[str, str]:
        """
        Performs the "Is it A or B?" reasoning.
        Returns the most resonant concept and a 'Causal Narrative' of the choice.
        """
        resonance_map = self.explore_possibilities(current_field, candidates)
        
        # Sort by resonance
        sorted_candidates = sorted(resonance_map.items(), key=lambda x: x[1], reverse=True)
        
        winner, win_res = sorted_candidates[0]
        runner_up, run_res = sorted_candidates[1] if len(sorted_candidates) > 1 else (winner, win_res)
        
        # Causal Logic: "I felt a stronger pull toward A than B"
        diff = win_res - run_res
        if diff > 0.1:
            narrative = f"Decisively chose {winner} over {runner_up} (Gradient: {diff:.2f})"
        else:
            narrative = f"Hesitantly manifesting {winner}, as it barely resonates more than {runner_up} (Gradient: {diff:.2f})"
            
        return winner, narrative
