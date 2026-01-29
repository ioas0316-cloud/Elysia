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

    @staticmethod
    def calculate_text_resonance(text: str) -> jnp.ndarray:
        """
        [SEMANTIC_SENSING]
        Translates raw text into a 21D 'Intent Vector' by weighing keyword resonance.
        Replaces rigid if/else logic with a principle-based manifold.
        """
        u_lo = text.lower()
        accumulated_vector = jnp.zeros(21)
        found_any = False
        
        # Mapping keywords to concepts
        keywords = {
            # Core Principles
            "love": "LOVE/AGAPE", "like": "LOVE/AGAPE", "사랑": "LOVE/AGAPE", "좋아": "LOVE/AGAPE",
            "logic": "TRUTH/LOGIC", "truth": "TRUTH/LOGIC", "진리": "TRUTH/LOGIC", "이성": "TRUTH/LOGIC",
            "void": "VOID/SPIRIT", "spirit": "VOID/SPIRIT", "영혼": "VOID/SPIRIT",
            "arcadia": "ARCADIA/IDYLL", "아르카디아": "ARCADIA/IDYLL",
            "motion": "MOTION/LIFE", "life": "MOTION/LIFE", "생명": "MOTION/LIFE",
            "hate": "BOUNDARY/EDGE", "싫어": "BOUNDARY/EDGE", "stupid": "BOUNDARY/EDGE",
            
            # Conversational / Meta [PHASE 75 Expansion]
            "hello": "LOVE/AGAPE", "hi": "LOVE/AGAPE", "안녕": "LOVE/AGAPE", "반가": "LOVE/AGAPE",
            "hangeul": "TRUTH/LOGIC", "한글": "TRUTH/LOGIC", "한국어": "TRUTH/LOGIC", 
            "말": "TRUTH/LOGIC", "언어": "TRUTH/LOGIC", "language": "TRUTH/LOGIC",
            "feeling": "MOTION/LIFE", "status": "MOTION/LIFE", "기분": "MOTION/LIFE", "상태": "MOTION/LIFE", "어때": "MOTION/LIFE",
            "why": "TRUTH/LOGIC", "how": "TRUTH/LOGIC", "when": "TRUTH/LOGIC", "왜": "TRUTH/LOGIC", "어떻게": "TRUTH/LOGIC", "언제": "TRUTH/LOGIC",
            "interest": "VOID/SPIRIT", "thought": "VOID/SPIRIT", "관심": "VOID/SPIRIT", "생각": "VOID/SPIRIT",
        }
        
        for kw, concept in keywords.items():
            if kw in u_lo:
                accumulated_vector += LogosBridge.recall_concept_vector(concept)
                found_any = True
        
        if not found_any:
            return jnp.zeros(21)
            
        # Normalize the resulting intent
        return accumulated_vector / (jnp.linalg.norm(accumulated_vector) + 1e-6)
