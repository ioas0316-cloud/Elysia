from typing import List, Dict, Tuple
import json
import os
from datetime import datetime
from Core.S1_Body.L6_Structure.Logic.trinary_logic import TrinaryLogic
from Core.S1_Body.L5_Mental.Reasoning.semantic_hypersphere import SemanticHypersphere
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignMath, SovereignVector
from Core.S1_Body.L6_Structure.M1_Merkaba.cognitive_terrain import CognitiveTerrain

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
    
    # [PHASE 160] Akashic Persistence Path
    AKASHIC_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))),
        "data", "S1_Body", "L5_Mental", "M1_Memory", "Raw", "Knowledge", "akashic_records.json"
    )
    
    # [PHASE 170] Cognitive Terrain for topological memory
    TERRAIN_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))),
        "maps", "cognitive_terrain.json"
    )
    TERRAIN = None  # Lazy initialization
    
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
        # Helper to extract real part from potentially complex values
        def real_val(v):
            return v.real if isinstance(v, complex) else v
        
        magnitude = real_val(vector.norm())
        if magnitude < 0.1: return "🌑 Void Mode"
        body_data = [real_val(x) for x in vector.data[:7]]
        soul_data = [real_val(x) for x in vector.data[7:14]]
        spirit_data = [real_val(x) for x in vector.data[14:]]
        body_mag = SovereignVector(body_data).norm()
        soul_mag = SovereignVector(soul_data).norm()
        spirit_mag = SovereignVector(spirit_data).norm()
        body_mag = real_val(body_mag)
        soul_mag = real_val(soul_mag)
        spirit_mag = real_val(spirit_mag)
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
        # Handle complex values by taking real part
        def real_val(v):
            return v.real if isinstance(v, complex) else v
        trits = [1 if real_val(v) > 0.5 else (-1 if real_val(v) < -0.5 else 0) for v in principle_vector.data]
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
        
        # [PHASE 160] Persist to Akashic Records (JSON fallback)
        LogosBridge._persist_to_akashic()
        
        # [PHASE 170] Carve into Cognitive Terrain (The River Carves the Land)
        LogosBridge._erode_terrain(u_name, vector)
    
    @staticmethod
    def _get_terrain():
        """Lazy initialization of CognitiveTerrain."""
        if LogosBridge.TERRAIN is None:
            try:
                LogosBridge.TERRAIN = CognitiveTerrain(map_file=LogosBridge.TERRAIN_PATH)
                print("🌍 [TERRAIN] Cognitive landscape connected.")
            except Exception as e:
                print(f"⚠️ [TERRAIN] Failed to load terrain: {e}")
                return None
        return LogosBridge.TERRAIN
    
    @staticmethod
    def _erode_terrain(concept_name: str, vector: SovereignVector):
        """[PHASE 170] The River Carves the Land - Learning shapes topology."""
        terrain = LogosBridge._get_terrain()
        if terrain is None:
            return
        
        try:
            # Map 21D vector to 2D terrain coordinates using first 2 dimensions
            # Normalize to terrain grid size
            real_data = [v.real if isinstance(v, complex) else v for v in vector.data[:2]]
            x = int((real_data[0] + 1) / 2 * (terrain.resolution - 1))  # [-1,1] -> [0, res-1]
            y = int((real_data[1] + 1) / 2 * (terrain.resolution - 1))
            x = max(0, min(x, terrain.resolution - 1))
            y = max(0, min(y, terrain.resolution - 1))
            
            # Inject as a prime keyword (creates an attractor/valley)
            magnitude = float(sum(abs(v.real if isinstance(v, complex) else v) for v in vector.data) / 21.0)
            terrain.inject_prime_keyword(x, y, concept_name, magnitude=magnitude * 0.5)
            
            print(f"🌊 [TERRAIN] '{concept_name}' carved at ({x}, {y}) with depth {magnitude:.2f}")
        except Exception as e:
            print(f"⚠️ [TERRAIN] Erosion failed: {e}")

    @staticmethod
    def _persist_to_akashic():
        """[PHASE 160] Save learned concepts to SSD (Akashic Records)."""
        try:
            data = {}
            for name, concept in LogosBridge.LEARNED_MAP.items():
                vec = concept["vector"]
                raw_data = vec.data if hasattr(vec, 'data') else list(vec)
                # Handle complex values by extracting real part
                clean_data = [float(v.real) if isinstance(v, complex) else float(v) for v in raw_data]
                data[name] = {
                    "vector": clean_data,
                    "description": concept.get("description", ""),
                    "stratum": concept["stratum"].value if hasattr(concept["stratum"], 'value') else concept["stratum"],
                    "timestamp": datetime.now().isoformat()
                }
            os.makedirs(os.path.dirname(LogosBridge.AKASHIC_PATH), exist_ok=True)
            with open(LogosBridge.AKASHIC_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"💾 [AKASHIC] Persisted {len(data)} concepts to eternal memory.")
        except Exception as e:
            print(f"⚠️ [AKASHIC] Persistence failed: {e}")

    @staticmethod
    def _load_from_akashic():
        """[PHASE 160] Restore learned concepts from SSD (Akashic Records)."""
        try:
            if not os.path.exists(LogosBridge.AKASHIC_PATH):
                return
            with open(LogosBridge.AKASHIC_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not data:
                return
            for name, concept in data.items():
                if name not in LogosBridge.LEARNED_MAP:
                    LogosBridge.LEARNED_MAP[name] = {
                        "vector": SovereignVector(concept["vector"]),
                        "description": concept.get("description", ""),
                        "stratum": MemoryStratum(concept.get("stratum", 0))
                    }
            print(f"📚 [AKASHIC] Restored {len(data)} concepts from eternal memory.")
        except Exception as e:
            print(f"⚠️ [AKASHIC] Restoration failed: {e}")

    @staticmethod
    def identify_concept(principle_vector: SovereignVector) -> Tuple[str, float]:
        best_concept = "UNKNOWN/CHAOS"
        max_resonance = -2.0
        for name, data in LogosBridge.CONCEPT_MAP.items():
            target = data["vector"]
            resonance = SovereignMath.resonance(principle_vector, target)
            # Handle complex resonance values (extract real part)
            if isinstance(resonance, complex):
                resonance = resonance.real
            weighted_resonance = float(resonance) * (1.0 + data["stratum"].value * 0.1)
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
