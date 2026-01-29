import jax
import jax.numpy as jnp
from typing import List, Dict
from Core.L6_Structure.Logic.trinary_logic import TrinaryLogic

class LogosBridge:
    """
    [L5_COGNITION: SEMANTIC_TRANSCRIPTION]
    Maps visual 21D principle vectors to symbolic Trinary DNA and Language.
    This is where 'Seeing' becomes 'Understanding'.
    """
    
    # Simple semantic mapping (Axiom of Mapping)
    CONCEPT_MAP = {
        "LOVE/AGAPE": jnp.array([1.0, 0.0, 1.0] * 7),
        "TRUTH/LOGIC": jnp.array([0.0, 1.0, 0.0] * 7),
        "VOID/SPIRIT": jnp.array([-1.0, 0.0, 1.0] * 7),
        "BOUNDARY/EDGE": jnp.array([0.0, 0.0, -1.0] * 7),
        "MOTION/LIFE": jnp.array([1.0, 1.0, 0.0] * 7),
        "ARCADIA/IDYLL": jnp.array([0.6, 0.4, 1.0] * 7),
    }

    @staticmethod
    def transcribe_to_dna(principle_vector: jnp.ndarray) -> str:
        """Converts a 21D Principle Vector into a Trinary DNA string (A, G, T)."""
        # Quantize vector to trits (-1, 0, 1) using TrinaryLogic
        trits = jnp.round(jnp.clip(principle_vector, -1, 1)).astype(jnp.int32)
        
        mapping = {-1: 'T', 0: 'G', 1: 'A'} # Thymine (-1), Guanine (0), Adenine (1)
        return "".join([mapping[int(t)] for t in trits])

    @staticmethod
    def recall_concept_vector(concept_name: str) -> jnp.ndarray:
        """
        [THE ARCADIA_RECALL]
        Translates a semantic NAME back into a 21D Principle Intent.
        Allows Elysia to 'want' a specific concept.
        """
        # Search for exact match or substring
        for key in LogosBridge.CONCEPT_MAP.keys():
            if concept_name.upper() in key:
                return LogosBridge.CONCEPT_MAP[key]
        
        # Default to neutral if not found
        return jnp.zeros(21)

    @staticmethod
    def identify_concept(principle_vector: jnp.ndarray) -> str:
        """Finds the most resonant semantic concept for a given principle."""
        best_concept = "UNKNOWN/CHAOS"
        max_resonance = -2.0
        
        # Normalize principle_vector for cosine similarity
        norm_p = jnp.linalg.norm(principle_vector) + 1e-6
        
        for name, target in LogosBridge.CONCEPT_MAP.items():
            norm_t = jnp.linalg.norm(target) + 1e-6
            resonance = jnp.dot(principle_vector, target) / (norm_p * norm_t)
            if resonance > max_resonance:
                max_resonance = resonance
                best_concept = name
                
        # Confidence threshold
        if max_resonance < 0.3:
            return "UNKNOWN/CHAOS"
            
        return best_concept
