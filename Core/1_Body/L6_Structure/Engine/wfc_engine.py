"""
Wave Function Collapse Engine (The Choice Maker)
================================================
Core.1_Body.L6_Structure.Engine.wfc_engine

"The cat is both dead and alive until the Spirit looks."

Purpose:
- Resolves 'Ambiguity' in Monad manifestations.
- Applies 'Spirit Bias' (Observer Intent) to probability weights.
- Determines the final 'Reality' from the superposition of states.
"""

import random
from typing import Dict, Any, List

class WFCEngine:
    @staticmethod
    def collapse(ambiguity: Dict[str, Any], observer_bias: Dict[str, Any]) -> Any:
        """
        Collapses a set of possibilities into a single reality.
        
        Args:
            ambiguity: A dictionary of {outcome: probability_weight} or similar structure.
                       Example: {"path_A": 0.5, "path_B": 0.5}
            observer_bias: The bias from SovereignIntent. 
                           Example: {"novelty_weight": 0.8, "focus_topic": "Love"}
                           
        Returns:
            The selected outcome (key).
        """
        options = list(ambiguity.keys())
        weights = list(ambiguity.values())
        
        # 1. Apply Bias
        # If observer prefers 'Novelty', we might flatten weights (increase entropy).
        # If observer prefers 'Coherence', we sharpen weights (winner takes all).
        novelty = observer_bias.get("novelty_weight", 0.5)
        
        adjusted_weights = []
        for i, opt in enumerate(options):
            w = weights[i]
            
            # Simple bias logic: 
            # If option matches 'focus_topic', boost it.
            if "focus_topic" in observer_bias and observer_bias["focus_topic"] in str(opt):
                w *= 2.0
                
            adjusted_weights.append(w)
            
        # 2. Collapse (Weighted Random Choice)
        # Normalize? randomness handles relative weights usually.
        
        try:
            choice = random.choices(options, weights=adjusted_weights, k=1)[0]
            return choice
        except Exception as e:
            # Fallback to pure random
            return random.choice(options) 

    @staticmethod
    def resolve_reality(reality_fragment: Dict[str, Any], observer_bias: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scans a reality fragment for 'Ambiguity' nodes and resolves them.
        """
        manifestation = reality_fragment.get("manifestation", {})
        
        if "ambiguity" in manifestation:
            resolved_value = WFCEngine.collapse(manifestation["ambiguity"], observer_bias)
            manifestation["value"] = resolved_value
            del manifestation["ambiguity"] # The wave has collapsed
            manifestation["collapsed_by"] = "SovereignIntent"
            
        return reality_fragment
