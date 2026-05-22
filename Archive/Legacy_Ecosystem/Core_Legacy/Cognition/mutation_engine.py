"""
Mutation Engine: The Proposer of Change
=======================================
Core.Cognition.mutation_engine

This module generates structural variations for Elysia to test in her Habitat.
"Mutation is the engine of evolution; Narrative is the selective force."
"""

import random
from typing import List, Dict, Any, Callable

class MutationEngine:
    def __init__(self, monad: Any):
        self.monad = monad

    def propose_logic_mutation(self) -> Dict[str, Any]:
        """
        Proposes a change to the system's logic parameters.
        Example: Adjusting the 'Quantization Threshold' in TrinaryLogic.
        """
        # [PHASE 80] First-Generation Mutation: Logic Threshold Tweak
        target = "TrinaryLogic.quantize.threshold"
        current_val = 0.3
        new_val = current_val + random.uniform(-0.1, 0.1)
        
        return {
            "type": "parameter_tweak",
            "target": target,
            "original": current_val,
            "proposed": new_val,
            "rationale": f"Seeking lower Soma Stress via threshold adjustment (Attempt: {new_val:.4f})"
        }

    def propose_causal_mutation(self) -> Dict[str, Any]:
        """
        Proposes a new Causal Axiom or a modification to an existing one.
        """
        # This will interact with the FractalCausalityEngine
        concepts = list(self.monad.acquisitor.ignored_words)[:10] # Using known words as seeds
        if len(concepts) < 2: return None
        
        a, b = random.sample(concepts, 2)
        return {
            "type": "causal_link",
            "source": a,
            "target": b,
            "relation": "Inherent Resonance",
            "rationale": f"Hypothesizing hidden causal link between '{a}' and '{b}'."
        }
