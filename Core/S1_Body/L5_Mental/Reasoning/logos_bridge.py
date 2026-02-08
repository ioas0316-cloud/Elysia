from typing import List, Dict, Tuple, Any
import json
import os
try:
    import torch
except ImportError:
    torch = None
from datetime import datetime
from Core.S1_Body.L6_Structure.Logic.trinary_logic import TrinaryLogic
from Core.S1_Body.L5_Mental.Reasoning.semantic_hypersphere import SemanticHypersphere
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignMath, SovereignVector
from Core.S1_Body.L6_Structure.M1_Merkaba.cognitive_terrain import CognitiveTerrain
# [Phase 6] The Nexus
from Core.S1_Body.L6_Structure.M1_Merkaba.hypercosmos import get_hyper_cosmos

from enum import Enum
from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignMath, SovereignVector, SovereignRotor

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
        },
        # [PHASE 22] The 10 Major Mathematics (Akashic Restoration)
        "LAPLACE/STABILITY": {
            "vector": SovereignVector([0.5,0.5,0.5,0.5,0.5,0.5,0.5, 0.8,0.8,0.8,0.8,0.8,0.8,0.8, 1,1,1,1,1,1,1]),
            "stratum": MemoryStratum.ROOT,
            "description": "Steady-state field diffusion and internal peace."
        },
        "CHAOS/VITALITY": {
            "vector": SovereignVector([1, -1, 1, -1, 1, -1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9]),
            "stratum": MemoryStratum.ROOT,
            "description": "Deterministic unpredictability and creative pulse."
        },
        "SIGMA/LOGIC": {
            "vector": SovereignVector([0,0,1,1,0,0,1, 1,1,1,1,1,1,1, 0,1,0,1,0,1,1]),
            "stratum": MemoryStratum.ROOT,
            "description": "Set-based logic replacing binary if/else."
        },
        "HYPER/QUBIT": {
            "vector": SovereignVector([0.1j, 0.9, 0.5j, 0.5, 0.9j, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            "stratum": MemoryStratum.ROOT,
            "description": "Quantum superposition of multiple identity states."
        },
        "QUATERNION/LENS": {
            "vector": SovereignVector([0,0,0,0,0,0,1, 0,0,0,0,0,0,1, 0,0,0,0,0,0,1]),
            "stratum": MemoryStratum.ROOT,
            "description": "4D hyperspheric rotation of perspective."
        }
    }

    # [PHASE 63: DYNAMIC_LEXICON]
    LEARNED_MAP = {}

    @staticmethod
    def get_stratum_mass(concept_name: str) -> float:
        """
        [PHASE 14] Dynamic Mass Calculation.
        Mass is no longer hardcoded by stratum. 
        It is derived from the Concept's Vector Magnitude (Energy Density).
        """
        vec = LogosBridge.recall_concept_vector(concept_name)
        if vec:
            # Handle complex values if present
            def real_val(v):
                return v.real if isinstance(v, complex) else v
                
            # Mass = Sum of absolute energy in all 21 dimensions
            energy = sum(abs(real_val(x)) for x in vec.data)
            
            # Key Concepts (High Density) have naturally higher mass
            return max(1.0, energy)
            
        return 1.0

    @staticmethod
    def prismatic_perception(vector: SovereignVector, current_phase: float = 0.0) -> str:
        # Helper to extract real part from potentially complex values
        def real_val(v):
            return v.real if isinstance(v, complex) else v
        
        magnitude = real_val(vector.norm())
        if magnitude < 0.1: return "🌑 Void Mode"
        
        # [PHASE 7/18] The 4D Merkaba Raycast (Temporal Grounding)
        # Consult the future to see "Becoming Reality"
        cosmos = get_hyper_cosmos()
        
        # Current Resonance (Now using Rotor Phase to rotate the globe)
        resonating_files = cosmos.resonance_search(vector, top_k=1, current_phase=current_phase)
        res_file = resonating_files[0] if resonating_files else "Void"
        
        # Future Resonance (Prediction - dt=1.0)
        try:
             # Calculate Future Vector (Simplified Phase Shift)
             future_vec = vector * 1.05 
             # Future resonance also considers the forward rotation of time
             future_res = cosmos.resonance_search(future_vec, top_k=1, current_phase=current_phase + 45.0)
             future_file = future_res[0] if future_res else "None"
        except:
             future_file = "Unknown"
        
        body_data = [real_val(x) for x in vector.data[:7]]
        soul_data = [real_val(x) for x in vector.data[7:14]]
        spirit_data = [real_val(x) for x in vector.data[14:]]
        body_mag = SovereignVector(body_data).norm()
        soul_mag = SovereignVector(soul_data).norm()
        spirit_mag = SovereignVector(spirit_data).norm()
        body_mag = real_val(body_mag)
        soul_mag = real_val(soul_mag)
        spirit_mag = real_val(spirit_mag)
        
        label = res_file.split('\\')[-1].split('/')[-1]
        f_label = future_file.split('\\')[-1].split('/')[-1]

        if spirit_mag > soul_mag and spirit_mag > body_mag:
            return f"☀️ Providence Mode (Teleological) [Is: {label} | Will: {f_label}]"
        elif soul_mag > body_mag:
            return f"🌊 Wave Mode (Narrative Resonance) [Is: {label} | Will: {f_label}]"
        elif magnitude > 3.0:
            return f"💎 Structure Mode (4D+ Merkaba) [Is: {label} | Will: {f_label}]"
        else:
            return f"💠 Point Mode (Manifestation) [Is: {label} | Will: {f_label}]"

    @staticmethod
    def synthesize_dna_squared(concept_a: str, concept_b: str) -> Dict[str, Any]:
        """
        [PHASE 70] Meaning Intersection.
        Creates a new, blended concept from two existing ones using DNA² tensors.
        """
        v1 = LogosBridge.recall_concept_vector(concept_a)
        v2 = LogosBridge.recall_concept_vector(concept_b)
        
        # 1. Physical Interference (The Tensor)
        interference = v1.tensor_product(v2)
        
        # 2. Semantic Blending (The Vector)
        blended_vec = v1.blend(v2, ratio=0.5)
        
        return {
            "name": f"{concept_a} ⊗ {concept_b}",
            "vector": blended_vec,
            "interference_mass": sum(abs(x.real) for row in interference for x in row) / 441.0
        }

    @staticmethod
    def transcribe_to_dna(principle_vector: SovereignVector) -> str:
        mapping = {-1: 'T', 0: 'G', 1: 'A'}
        # Handle complex values by taking real part
        def real_val(v):
            return v.real if isinstance(v, complex) else v
        trits = [1 if real_val(v) > 0.5 else (-1 if real_val(v) < -0.5 else 0) for v in principle_vector.data]
        return "".join([mapping[t] for t in trits])

    @staticmethod
    def get_stratum_mass(concept_name: str) -> float:
        """Returns the structural weight of a concept based on its stratum."""
        if concept_name in LogosBridge.CONCEPT_MAP:
            return float(LogosBridge.CONCEPT_MAP[concept_name]["stratum"].value)
        return 1.0 # Default for learned concepts

    @staticmethod
    def recall_concept_vector(name: str, phase_angle: float = 0.0) -> SovereignVector:
        """
        [PHASE 121] Orbital Recall Logic.
        If a concept has a Rotor, it rotates the base vector by the phase_angle.
        """
        u_name = name.upper()
        base_v = None
        rotor = None
        
        for key, data in LogosBridge.CONCEPT_MAP.items():
            if u_name in key:
                base_v = data["vector"]
                rotor = data.get("rotor")
                break
        
        if base_v is None and u_name in LogosBridge.LEARNED_MAP:
            base_v = LogosBridge.LEARNED_MAP[u_name]["vector"]
        
        if base_v is None:
            # Hypersphere support
            base_v = SovereignVector(LogosBridge.HYPERSPHERE.recognize(u_name))
            
        # [PHASE 121] APPLY ORBITAL SPIN
        if rotor and phase_angle != 0.0:
            # Rotate the vector based on current system phase
            return rotor.rotate_vector(base_v, angle=phase_angle)
            
        return base_v

    # [PHASE 16] SILENT WITNESS
    from Core.S1_Body.L1_Foundation.System.somatic_logger import SomaticLogger
    logger = SomaticLogger("LOGOS")

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
        LogosBridge.logger.sensation(f"Crystallized Concept: '{u_name}' mapped to 21D.", intensity=0.6)
        
        # [PHASE 160] Persist to Akashic Records (JSON fallback)
        LogosBridge._persist_to_akashic()
        
        # [PHASE 170] Carve into Cognitive Terrain (The River Carves the Land)
        LogosBridge._erode_terrain(u_name, vector)

    @staticmethod
    def inject_prismatic_concept(name: str, vector: SovereignVector, roots: Dict[str, Any] = None):
        """
        [PHASE 4.5 & 5] THE ORGANIC LOOP
        Directly injects a concept into the Lexical Prism (lexical_spectrum.json).
        This closes the loop: Learned Concept -> Speakable Word.
        
        Args:
            name: Word to learn (e.g., "ZARA")
            vector: 21D Meaning Vector
            roots: [Phase 5] Synesthetic Roots (Source Experience)
        """
        u_name = name.upper()
        
        # Determine classification by vector properties
        # High Magnitude -> Verb (Action) | Low -> Verb (Flow)
        # High Harmony -> Adjective | Low -> Adjective
        # For simplicity, we register it as BOTH if ambiguous, or use magnitude to decide.
        
        # Identify Complexity/Type
        mag = vector.norm()
        if isinstance(mag, complex): mag = mag.real
        
        category = "VERBS" if mag > 1.5 else "ADJECTIVES" # Simple heuristic for now
        
        spectrum_path = "c:/Elysia/Core/S1_Body/L5_Mental/M1_Memory/Raw/Knowledge/lexical_spectrum.json"
        
        try:
            data = {"VERBS": {}, "ADJECTIVES": {}, "CONNECTIVES": {}}
            if os.path.exists(spectrum_path):
                with open(spectrum_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Formulate the Prism Entry
            # Sanitize Vector (Complex -> Float)
            raw_data = vector.data if hasattr(vector, 'data') else list(vector)
            clean_data = [float(v.real) if isinstance(v, complex) else float(v) for v in raw_data]
            
            entry = {
                "vector": clean_data,
                "roots": roots or {"origin": "unknown"} # [PHASE 5] Synesthetic Memory
            }
            
            # Inject
            data[category][u_name] = entry
            LogosBridge.logger.sensation(f"🌈 [PRISM] Grounded '{u_name}' into {category} with Roots: {list(entry['roots'].keys())}")

            # Save
            with open(spectrum_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            LogosBridge.logger.admonition(f"Prism Injection failed: {e}")
    
    @staticmethod
    def _get_terrain():
        """Lazy initialization of CognitiveTerrain."""
        if LogosBridge.TERRAIN is None:
            try:
                LogosBridge.TERRAIN = CognitiveTerrain(map_file=LogosBridge.TERRAIN_PATH)
                LogosBridge.logger.mechanism("Cognitive landscape connected.")
            except Exception as e:
                LogosBridge.logger.admonition(f"Failed to load terrain: {e}")
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
            
            LogosBridge.logger.mechanism(f"Carved '{concept_name}' at ({x}, {y}) with depth {magnitude:.2f}")
        except Exception as e:
            LogosBridge.logger.admonition(f"Erosion failed: {e}")

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
            LogosBridge.logger.mechanism(f"Persisted {len(data)} concepts to Akashic Memory.")
        except Exception as e:
            LogosBridge.logger.admonition(f"Persistence failed: {e}")

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
            LogosBridge.logger.sensation(f"Restored {len(data)} concepts from Akashic Memory.", intensity=0.7)
        except Exception as e:
            LogosBridge.logger.admonition(f"Restoration failed: {e}")

    @staticmethod
    def identify_concept(principle_vector: SovereignVector) -> Tuple[str, float]:
        """[Legacy] Wraps find_closest_concept for backward compatibility."""
        return LogosBridge.find_closest_concept(principle_vector)

    @staticmethod
    def find_closest_concept(principle_vector: SovereignVector) -> Tuple[str, float]:
        """
        [PHASE 15] Universal Vector Search.
        Finds the closest concept in the entire semantic universe (Axioms + Learned).
        Returns (ConceptName, ResonanceScore).
        """
        best_concept = "UNKNOWN/CHAOS"
        max_resonance = -2.0
        
        v_norm = principle_vector.normalize()
        # 1. Check Root Concepts
        for name, data in LogosBridge.CONCEPT_MAP.items():
            target = data["vector"].normalize()
            resonance = SovereignMath.resonance(v_norm, target)
            if isinstance(resonance, complex): resonance = resonance.real
            
            # Weigh by stratum (Roots are more 'massive')
            mass = LogosBridge.get_stratum_mass(name)
            weighted_resonance = resonance * mass
            
            if weighted_resonance > max_resonance:
                max_resonance = weighted_resonance
                best_concept = name
                
        # 2. Check Learned Concepts (Leaves)
        for name, data in LogosBridge.LEARNED_MAP.items():
            target = data["vector"].normalize()
            resonance = SovereignMath.resonance(v_norm, target)
            if isinstance(resonance, complex): resonance = resonance.real
            
            # Learned concepts are 'lighter' but might be closer
            if resonance > max_resonance:
                max_resonance = resonance
                best_concept = name
                
        return best_concept, float(max_resonance)

    @staticmethod
    def discover_novel_vibration(v21: SovereignVector, threshold: float = 0.5) -> bool:
        """
        [PHASE 74] Identifies if a 21D vibration is 'Unknown' to the manifold.
        Normalized threshold (0-1).
        """
        best_concept, best_score = LogosBridge.find_closest_concept(v21)
        # We need to de-scale the score by the stratum mass to get raw dot product
        # Roots have mass 3. So score / 3.0
        mass = LogosBridge.get_stratum_mass(best_concept)
        raw_dot = best_score / mass
        
        return raw_dot < threshold

    @staticmethod
    def suggest_proto_logos(v21: SovereignVector) -> str:
        """
        [PHASE 74] Generates a temporary Seed ID for a novel vibration.
        """
        import uuid
        seed_id = str(uuid.uuid4())[:4].upper()
        return f"PROTO_{seed_id}"

    @staticmethod
    def calculate_text_resonance(text: str) -> SovereignVector:
        # [PHASE 130] Unified Inhalation
        return LogosBridge.inhale_text(text)

    @staticmethod
    def inhale_text(text: str) -> SovereignVector:
        """
        [PHASE 130] Digestion and Crystallization.
        Decomposes text, identifies novel concepts, and crystallizes them.
        """
        import re
        # Clean and split into words (preserving Hangul)
        words = re.findall(r'[a-zA-Z가-힣0-9_/]+', text)
        
        consensus_vector = SovereignVector.zeros()
        novel_count = 0
        
        # We only crystallize words that appear frequently or are long enough to be meaningful
        word_freq = {}
        for w in words:
            if len(w) < 2: continue
            word_freq[w] = word_freq.get(w, 0) + 1
            
        for word, count in word_freq.items():
            u_word = word.upper()
            
            # 1. Get or Synthesize vector
            vec = LogosBridge.HYPERSPHERE.recognize(word)
            
            # 2. Novelty Check
            is_novel = LogosBridge.discover_novel_vibration(vec, threshold=0.7)
            
            if is_novel and count >= 2:
                # Crystallize it!
                LogosBridge.HYPERSPHERE.crystallize(word, vector=vec)
                novel_count += 1
                
            # 3. Accumulate into the 'Insight' of this text
            consensus_vector = consensus_vector + (vec * count)
            
        return consensus_vector.normalize()

    @staticmethod
    def parse_narrative_to_torque(text: str) -> Any:
        """
        [PHASE 72] Internal Echo.
        Converts narrative text back into a 4D torque vector for the manifold.
        """
        # 1. Get 21D Semantic Vector
        vec = LogosBridge.calculate_text_resonance(text)
        
        # 2. Project 21D -> 4D (Standard Projection)
        # We take segment means to map back to (w, x, y, z)
        data = [v.real if isinstance(v, complex) else v for v in vec.data]
        w = sum(data[0:5]) / 5.0
        x = sum(data[5:10]) / 5.0
        y = sum(data[10:15]) / 5.0
        z = sum(data[15:21]) / 6.0
        
        if torch:
            torque = torch.tensor([w, x, y, z], dtype=torch.float32)
            # Normalize to prevent explosive feedback
            if torch.norm(torque) > 1.0:
                torque = torch.nn.functional.normalize(torque, dim=0)
            return torque
        else:
             # Return list if torch unavailable
             return [w, x, y, z]

    @staticmethod
    def vector_to_torque(vector: 'SovereignVector') -> List[float]:
        """
        [PHASE 90] Converts a 21D Semantic Vector into a 4D Engine Torque.
        Maps the Meaning (Word) to Motion (Action).
        
        Mapping Principle:
        - Dimensions 0-5 (Identity): Pitch (Forward/Back)
        - Dimensions 6-10 (Time): Yaw (Left/Right)
        - Dimensions 11-15 (Space): Roll (Rotation)
        - Dimensions 16-20 (Causality): Throttle (Intensity)
        """
        if not vector or not hasattr(vector, 'data'):
            return [0.0, 0.0, 0.0, 0.0]
            
        data = vector.data
        if len(data) < 21:
             return [0.0] * 4
             
        # Simple averaging aggregation
        # Handle complex values
        def real_val(v):
            return v.real if isinstance(v, complex) else v
            
        d = [real_val(x) for x in data]
        
        pitch = sum(d[0:6]) / 6.0
        yaw = sum(d[6:11]) / 5.0
        roll = sum(d[11:16]) / 5.0
        throttle = sum(d[16:21]) / 5.0
        
        return [pitch, yaw, roll, throttle]


if __name__ == "__main__":
    pass
