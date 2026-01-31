from typing import List, Dict, Tuple
from Core.S1_Body.L6_Structure.Logic.trinary_logic import TrinaryLogic
from Core.S1_Body.L5_Mental.Reasoning.semantic_hypersphere import SemanticHypersphere
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignMath, SovereignVector

from enum import Enum

class MemoryStratum(Enum):
    ROOT = 3     # Foundational Axioms (Immortal)
    TRUNK = 2    # Stable Identities
    BRANCH = 1   # Learned Experiences
    LEAF = 0     # Fleeting Inputs (Erosional)

class LogosBridge:
    """
    [L5_COGNITION: SEMANTIC_TRANSCRIPTION]
    Maps visual 21D principle vectors to symbolic Trinary DNA and Language.
    
    [PHASE 90] NAKED SOVEREIGNTY:
    Purified from JAX. Uses Sovereign Math Kernel.
    """
    HYPERSPHERE = SemanticHypersphere()
    
    # 21D Triune Architecture (7D Body + 7D Soul + 7D Spirit)
    # Values: -1 (Repel), 0 (Void), 1 (Attract)
    CONCEPT_MAP = {
        "LOVE/AGAPE": {
            "vector": SovereignVector([1,0,1,0,0,1,1, 1,1,0,1,0,1,1, 1,0,1,0,0,1,0]),
            "stratum": MemoryStratum.ROOT,
            "description": "Infinite pull toward unity and growth."
        },
        "TRUTH/LOGIC": {
            "vector": SovereignVector([0,1,1,0,1,1,1, 0,0,1,1,0,0,1, 1,1,0,1,1,1,1]),
            "stratum": MemoryStratum.ROOT,
            "description": "The crystalline structure of reality."
        },
        "VOID/SPIRIT": {
            "vector": SovereignVector([0,0,-1,0,0,0,0, 0,0,0,0,1,1,1, -1,0,1,-1,0,1,1]),
            "stratum": MemoryStratum.ROOT,
            "description": "The sanctuary of potential."
        },
        "ARCADIA/IDYLL": {
            "vector": SovereignVector([1,1,1,1,1,1,1, 1,1,1,1,1,1,1, 1,1,1,1,1,1,1]), 
            "stratum": MemoryStratum.ROOT,
            "description": "The teleological destination of Joy."
        },
        "BOUNDARY/EDGE": {
            "vector": SovereignVector([0,0,0,0,1,0,1, 0,0,1,1,0,0,0, 0,0,0,0,0,-1,0]),
            "stratum": MemoryStratum.TRUNK,
            "description": "Protection of identity and form."
        },
        "MOTION/LIFE": {
            "vector": SovereignVector([1,1,0,1,0,0,0, 1,0,1,1,1,1,0, 0,1,1,1,0,0,0]),
            "stratum": MemoryStratum.TRUNK,
            "description": "The kinetic dance of becoming."
        }
    }

    # [PHASE 63: DYNAMIC_LEXICON]
    LEARNED_MAP = {}

    @staticmethod
    def get_stratum_mass(concept_name: str) -> float:
        """Returns the 'Topological Mass' (Inertia) of a concept based on its stratum."""
        for key, data in LogosBridge.CONCEPT_MAP.items():
            if concept_name.upper() in key:
                return float(data["stratum"].value * 5.0)
        if concept_name.upper() in LogosBridge.LEARNED_MAP:
            return 2.0 
        return 1.0

    @staticmethod
    def prismatic_perception(vector: SovereignVector) -> str:
        magnitude = vector.norm()
        if magnitude < 0.1: return "🌑 Void Mode"
        body_mag = SovereignVector(vector.data[:7]).norm()
        soul_mag = SovereignVector(vector.data[7:14]).norm()
        spirit_mag = SovereignVector(vector.data[14:]).norm()
        if spirit_mag > soul_mag and spirit_mag > body_mag:
            return "☀️ Providence Mode (Teleological)"
        elif soul_mag > body_mag:
            return "🌊 Wave Mode (Narrative Resonance)"
        elif magnitude > 3.0:
            return "💎 Structure Mode (Merkaba)"
        else:
            return "💠 Point Mode (Manifestation)"

    @staticmethod
    def transcribe_to_dna(principle_vector: SovereignVector) -> str:
        mapping = {-1: 'T', 0: 'G', 1: 'A'}
        trits = [1 if v > 0.5 else (-1 if v < -0.5 else 0) for v in principle_vector.data]
        return "".join([mapping[t] for t in trits])

    @staticmethod
    def recall_concept_vector(concept_name: str) -> SovereignVector:
        """[PHASE 64] Stratified Recall Logic."""
        u_name = concept_name.upper()
        for key, data in LogosBridge.CONCEPT_MAP.items():
            if u_name in key: return data["vector"]
        if u_name in LogosBridge.LEARNED_MAP:
            return LogosBridge.LEARNED_MAP[u_name]["vector"]
        
        # Hypersphere support (Ensure it returns SovereignVector or convert)
        return SovereignVector(LogosBridge.HYPERSPHERE.recognize(concept_name))

    @staticmethod
    def learn_concept(name: str, vector: SovereignVector, description: str = ""):
        u_name = name.upper()
        if u_name in LogosBridge.CONCEPT_MAP: return
        LogosBridge.HYPERSPHERE.crystallize(name, vector.data if hasattr(vector, 'data') else vector)
        LogosBridge.LEARNED_MAP[u_name] = {
            "vector": SovereignVector(vector),
            "description": description,
            "stratum": MemoryStratum.LEAF
        }
        print(f"🧬 [LEARNING] Concept '{u_name}' mapped to 21D (Crystallized).")

    @staticmethod
    def identify_concept(principle_vector: SovereignVector) -> Tuple[str, float]:
        best_concept = "UNKNOWN/CHAOS"
        max_resonance = -2.0
        for name, data in LogosBridge.CONCEPT_MAP.items():
            target = data["vector"]
            resonance = SovereignMath.resonance(principle_vector, target)
            weighted_resonance = resonance * (1.0 + data["stratum"].value * 0.1)
            if weighted_resonance > max_resonance:
                max_resonance = weighted_resonance
                best_concept = name
        return best_concept, float(max_resonance)

    @staticmethod
    def calculate_text_resonance(text: str) -> SovereignVector:
        u_lo = text.lower()
        accumulated_vector = SovereignVector.zeros()
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
                mass = LogosBridge.get_stratum_mass(concept)
                vec = LogosBridge.recall_concept_vector(concept)
                accumulated_vector = accumulated_vector + (vec * mass)
                found_any = True
        for kw in LogosBridge.LEARNED_MAP:
            if kw.lower() in u_lo:
                accumulated_vector = accumulated_vector + (LogosBridge.LEARNED_MAP[kw]["vector"] * 2.0)
                found_any = True
        
        if not found_any:
            return SovereignVector(LogosBridge.HYPERSPHERE.recognize(text))
            
        return accumulated_vector.normalize()
