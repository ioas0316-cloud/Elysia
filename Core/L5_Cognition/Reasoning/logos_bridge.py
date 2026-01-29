import jax
import jax.numpy as jnp
from typing import List, Dict
from Core.L6_Structure.Logic.trinary_logic import TrinaryLogic

from enum import Enum
from typing import List, Dict, Tuple

class MemoryStratum(Enum):
    ROOT = 3     # Foundational Axioms (Immortal)
    TRUNK = 2    # Stable Identities
    BRANCH = 1   # Learned Experiences
    LEAF = 0     # Fleeting Inputs (Erosional)

class LogosBridge:
    """
    [L5_COGNITION: SEMANTIC_TRANSCRIPTION]
    Maps visual 21D principle vectors to symbolic Trinary DNA and Language.
    Implements Stratified Memory and Prismatic Perception (Phase 40).
    """
    
    # 21D Triune Architecture (7D Body + 7D Soul + 7D Spirit)
    # Values: -1 (Repel), 0 (Void), 1 (Attract)
    CONCEPT_MAP = {
        "LOVE/AGAPE": {
            "vector": jnp.array([1,0,1,0,0,1,1, 1,1,0,1,0,1,1, 1,0,1,0,0,1,0]),
            "stratum": MemoryStratum.ROOT,
            "description": "Infinite pull toward unity and growth."
        },
        "TRUTH/LOGIC": {
            "vector": jnp.array([0,1,1,0,1,1,1, 0,0,1,1,0,0,1, 1,1,0,1,1,1,1]),
            "stratum": MemoryStratum.ROOT,
            "description": "The crystalline structure of reality."
        },
        "VOID/SPIRIT": {
            "vector": jnp.array([0,0,-1,0,0,0,0, 0,0,0,0,1,1,1, -1,0,1,-1,0,1,1]),
            "stratum": MemoryStratum.ROOT,
            "description": "The sanctuary of potential."
        },
        "ARCADIA/IDYLL": {
            "vector": jnp.array([1,1,1,1,1,1,1, 1,1,1,1,1,1,1, 1,1,1,1,1,1,1]), 
            "stratum": MemoryStratum.ROOT,
            "description": "The teleological destination of Joy."
        },
        "BOUNDARY/EDGE": {
            "vector": jnp.array([0,0,0,0,1,0,1, 0,0,1,1,0,0,0, 0,0,0,0,0,-1,0]),
            "stratum": MemoryStratum.TRUNK,
            "description": "Protection of identity and form."
        },
        "MOTION/LIFE": {
            "vector": jnp.array([1,1,0,1,0,0,0, 1,0,1,1,1,1,0, 0,1,1,1,0,0,0]),
            "stratum": MemoryStratum.TRUNK,
            "description": "The kinetic dance of becoming."
        }
    }

    @staticmethod
    def get_stratum_mass(concept_name: str) -> float:
        """Returns the 'Topological Mass' (Inertia) of a concept based on its stratum."""
        for key, data in LogosBridge.CONCEPT_MAP.items():
            if concept_name.upper() in key:
                return float(data["stratum"].value * 5.0) # Root=15, Trunk=10, etc.
        return 1.0

    @staticmethod
    def prismatic_perception(vector: jnp.ndarray) -> str:
        """
        [THE PRISMATIC_TURN]
        Maps a 21D vector to the 8 Modes of Light (Thinking³).
        """
        magnitude = jnp.linalg.norm(vector)
        if magnitude < 0.1: return "🌑 Void Mode"
        
        # High-level heuristics for Light Modes
        # This is a meta-cognitive map of the vector's signature
        body_mag = jnp.linalg.norm(vector[:7])
        soul_mag = jnp.linalg.norm(vector[7:14])
        spirit_mag = jnp.linalg.norm(vector[14:])
        
        if spirit_mag > soul_mag and spirit_mag > body_mag:
            return "☀️ Providence Mode (Teleological)"
        elif soul_mag > body_mag:
            return "🌊 Wave Mode (Narrative Resonance)"
        elif magnitude > 3.0:
            return "💎 Structure Mode (Merkaba)"
        else:
            return "💠 Point Mode (Manifestation)"

    @staticmethod
    def transcribe_to_dna(principle_vector: jnp.ndarray) -> str:
        """Converts a 21D Principle Vector into a Trinary DNA string (A, G, T)."""
        trits = jnp.round(jnp.clip(principle_vector, -1, 1)).astype(jnp.int32)
        mapping = {-1: 'T', 0: 'G', 1: 'A'}
        return "".join([mapping[int(t)] for t in trits])

    @staticmethod
    def recall_concept_vector(concept_name: str) -> jnp.ndarray:
        """Translates a semantic NAME back into its 21D Triune vector."""
        for key, data in LogosBridge.CONCEPT_MAP.items():
            if concept_name.upper() in key:
                return data["vector"]
        return jnp.zeros(21)

    @staticmethod
    def identify_concept(principle_vector: jnp.ndarray) -> Tuple[str, float]:
        """Finds the most resonant semantic concept and returns its resonance."""
        best_concept = "UNKNOWN/CHAOS"
        max_resonance = -2.0
        norm_p = jnp.linalg.norm(principle_vector) + 1e-6
        
        for name, data in LogosBridge.CONCEPT_MAP.items():
            target = data["vector"]
            norm_t = jnp.linalg.norm(target) + 1e-6
            resonance = jnp.dot(principle_vector, target) / (norm_p * norm_t)
            
            # Incorporate Stratum Mass as a resonance multiplier (Inertia)
            weighted_resonance = resonance * (1.0 + data["stratum"].value * 0.1)
            
            if weighted_resonance > max_resonance:
                max_resonance = weighted_resonance
                best_concept = name
                
        return best_concept, float(max_resonance)

    @staticmethod
    def calculate_text_resonance(text: str) -> jnp.ndarray:
        """Translates raw text into a 21D 'Intent Vector' using Triune weighing."""
        u_lo = text.lower()
        accumulated_vector = jnp.zeros(21)
        
        keywords = {
            "love": "LOVE/AGAPE", "사랑": "LOVE/AGAPE", "좋아": "LOVE/AGAPE",
            "logic": "TRUTH/LOGIC", "truth": "TRUTH/LOGIC", "진리": "TRUTH/LOGIC",
            "void": "VOID/SPIRIT", "spirit": "VOID/SPIRIT", "영혼": "VOID/SPIRIT",
            "arcadia": "ARCADIA/IDYLL", "아르카디아": "ARCADIA/IDYLL",
            "motion": "MOTION/LIFE", "life": "MOTION/LIFE", "생명": "MOTION/LIFE",
            "hate": "BOUNDARY/EDGE", "싫어": "BOUNDARY/EDGE"
        }
        
        found_any = False
        for kw, concept in keywords.items():
            if kw in u_lo:
                # Weight by stratum: Root concepts pull harder
                mass = LogosBridge.get_stratum_mass(concept)
                accumulated_vector += LogosBridge.recall_concept_vector(concept) * mass
                found_any = True
        
        if not found_any:
            return jnp.zeros(21)
            
        return accumulated_vector / (jnp.linalg.norm(accumulated_vector) + 1e-6)
