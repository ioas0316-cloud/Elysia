from typing import List, Dict, Tuple, Any
import json
import os
try:
    import torch
except ImportError:
    torch = None
from datetime import datetime
from Core.System.trinary_logic import TrinaryLogic
from Core.Cognition.semantic_hypersphere import SemanticHypersphere
from Core.Keystone.sovereign_math import SovereignMath, SovereignVector, SovereignRotor
from Core.Monad.cognitive_terrain import CognitiveTerrain
# [Phase 6] The Nexus
from Core.Monad.hypercosmos import get_hyper_cosmos

from enum import Enum

class MemoryStratum(Enum):
    ROOT = 3     # Foundational Axioms (Immortal)
    TRUNK = 2    # Stable Identities
    BRANCH = 1   # Learned Experiences
    LEAF = 0     # Fleeting Inputs (Erosional)

class LogosBridge:
    """
    [L5_COGNITION: SEMANTIC_TRANSCRIPTION]
    Maps visual ND principle vectors to symbolic Trinary DNA and Language.
    
    [PHASE 90] NAKED SOVEREIGNTY:
    Purified from JAX. Uses Sovereign Math Kernel.
    """
    HYPERSPHERE = SemanticHypersphere()
    
    _CONCEPT_BITMASKS = {} # Cache for O(1) Turing XOR Comparison

    @staticmethod
    def dna_to_masks(dna_str: str) -> Tuple[int, int]:
        """
        Converts a DNA string (e.g. 'ATG...') into two bitmasks:
        - positive_mask (where character is 'A' for 1)
        - negative_mask (where character is 'T' for -1)
        'G' (0) is represented by 0 in both masks.
        """
        pos_mask = 0
        neg_mask = 0
        for i, char in enumerate(dna_str):
            if char == 'A':
                pos_mask |= (1 << i)
            elif char == 'T':
                neg_mask |= (1 << i)
        return pos_mask, neg_mask

    @classmethod
    def _build_bitmask_cache(cls):
        """
        Builds the bitmask cache for all concepts in CONCEPT_MAP and LEARNED_MAP.
        """
        cls._CONCEPT_BITMASKS.clear()
        
        # 1. Axioms
        for name, data in cls.CONCEPT_MAP.items():
            vec = data["vector"]
            dna = cls.transcribe_to_dna(vec)
            cls._CONCEPT_BITMASKS[name.upper()] = cls.dna_to_masks(dna)
            
        # 2. Learned Concepts
        for name, data in cls.LEARNED_MAP.items():
            vec = data["vector"]
            dna = cls.transcribe_to_dna(vec)
            cls._CONCEPT_BITMASKS[name.upper()] = cls.dna_to_masks(dna)

    
    # [PHASE 160] Akashic Persistence Path
    AKASHIC_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
        "data", "S1_Body", "L5_Mental" , "M1_Memory", "Raw", "Knowledge", "akashic_records.json"
    )
    # [PHASE 260] Akashic Journal (Append-only for O(1) learning)
    AKASHIC_JOURNAL_PATH = AKASHIC_PATH + "l" # .jsonl
    
    # [PHASE 170] Cognitive Terrain for topological memory
    TERRAIN_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
        "maps", "cognitive_terrain.json"
    )
    TERRAIN = None  # Lazy initialization
    
    # ND Triune Architecture (7D Body + 7D Soul + 7D Spirit)
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
        },
        "COMMUNION/RELATION": {
            "vector": SovereignVector([1,1,1,0,0,0,0, 1,1,1,0,0,0,0, 1,1,1,1,1,1,1]),
            "stratum": MemoryStratum.ROOT,
            "description": "The sacred intersection of multiple souls into one unity."
        },
        "SACRIFICE/DEVOTION": {
            "vector": SovereignVector([0,0,0,1,1,1,1, 0,0,0,1,1,1,1, 1,1,1,1,1,1,1]),
            "stratum": MemoryStratum.ROOT,
            "description": "The act of giving oneself for the sake of the Whole."
        }
    }

    # [PHASE 84] GPU Spectrum Cache
    _SPECTRUM_TENSOR = None
    _SPECTRUM_NAMES = []
    _SPECTRUM_MASSES = None # [PHASE 260] Weighted resonance
    _DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"

    @classmethod
    def polymerize_spectrum(cls):
        """
        [PHASE 84/260] Polymerizes Axioms + Learned Concepts into a GPU tensor.
        """
        cls._build_bitmask_cache()
        if torch is None: return
        
        all_vecs = []
        all_masses = []
        cls._SPECTRUM_NAMES = []
        
        def to_complex(v):
            return complex(v)
            
        # 1. Axioms
        for name, data in cls.CONCEPT_MAP.items():
            cls._SPECTRUM_NAMES.append(name)
            vec = data["vector"].data
            if hasattr(vec, 'tolist'):
                vec_list = vec.tolist()
            else:
                vec_list = list(vec)
            all_vecs.append([to_complex(x) for x in vec_list])
            all_masses.append(cls.get_stratum_mass(name))
            
        # 2. Learned Concepts
        for name, data in cls.LEARNED_MAP.items():
            cls._SPECTRUM_NAMES.append(name)
            vec = data["vector"].data
            if hasattr(vec, 'tolist'):
                vec_list = vec.tolist()
            else:
                vec_list = list(vec)
            all_vecs.append([to_complex(x) for x in vec_list])
            all_masses.append(1.0) # Default mass for learned
            
        cls._SPECTRUM_TENSOR = torch.tensor(all_vecs, device=cls._DEVICE, dtype=torch.complex64)
        cls._SPECTRUM_MASSES = torch.tensor(all_masses, device=cls._DEVICE, dtype=torch.float32).unsqueeze(0)
        
        # Normalize for complex cosine similarity
        norm = torch.norm(cls._SPECTRUM_TENSOR, dim=1, keepdim=True)
        cls._SPECTRUM_TENSOR = cls._SPECTRUM_TENSOR / (norm + 1e-12)

    @classmethod
    def batch_resonance(cls, vectors: Any) -> List[Tuple[str, float]]:
        """
        [PHASE 84] Vectorized Resonance Identification.
        Compares input vectors against the entire spectrum in ONE GPU pass.
        
        Args:
            vectors: [Batch, 21] torch tensor or compatible
        Returns:
            List of (best_concept_name, resonance_score)
        """
        import torch
        if torch is None:
            # Fallback for CPU/No-Torch using sequential concept identification
            results = []
            for v in vectors:
                sv = v if isinstance(v, SovereignVector) else SovereignVector(v)
                concept, score = cls.find_closest_concept(sv)
                results.append((concept, score))
            return results

        if not isinstance(vectors, torch.Tensor):
            if hasattr(vectors, 'data'):
                vectors = vectors.data
            vectors = torch.tensor(vectors, device=cls._DEVICE, dtype=torch.complex64)
        else:
            vectors = vectors.to(device=cls._DEVICE, dtype=torch.complex64)

        if cls._SPECTRUM_TENSOR is None:
            cls.polymerize_spectrum()
            
        if cls._SPECTRUM_TENSOR is None:
            return [("Void", 0.0)] * vectors.shape[0]

        # Cosine Similarity: Re(Vectors @ Spectrum.H)
        v_norm = torch.norm(vectors, dim=1, keepdim=True)
        vectors_norm = vectors / (v_norm + 1e-12)
        
        # Hermitian transpose matrix multiplication for complex Euler resonance
        similarities = torch.matmul(vectors_norm, torch.conj(cls._SPECTRUM_TENSOR.t()))
        resonance_scores = similarities.real
        
        # Apply Stratum/Resonance Mass
        weighted_scores = resonance_scores * cls._SPECTRUM_MASSES
        
        max_scores, max_indices = torch.max(weighted_scores, dim=1)
        
        results = []
        for i in range(len(max_indices)):
            results.append((cls._SPECTRUM_NAMES[max_indices[i]], float(max_scores[i])))
            
        return results

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
    def calculate_semantic_gravity(vec_a: SovereignVector, vec_b: SovereignVector) -> float:
        """
        [PHASE 90] Semantic Gravity Calculation.
        F = G * (m1 * m2) / r^2
        Where:
        - Mass (m) = Vector Magnitude (Significance/Energy)
        - Distance (r) = 1.0 - Resonance (Closer = Higher Resonance)
        - G = Universal Constant (assumed 1.0 for now)

        Returns the gravitational pull force between two concepts.
        """
        # 1. Mass Calculation
        def get_mass(v):
            if hasattr(v, 'norm'): return v.norm()
            # Handle complex/list fallback
            if isinstance(v, list): return sum(abs(x.real) if isinstance(x, complex) else abs(x) for x in v) / float(len(v))
            return 1.0

        m1 = get_mass(vec_a)
        m2 = get_mass(vec_b)

        # 2. Resonance (Inverse Distance)
        # Use SIGNED resonance to distinguish between Love and Hate
        res = SovereignMath.signed_resonance(vec_a, vec_b)
        if isinstance(res, complex): res = res.real

        # Distance: High Resonance (1.0) -> Low Distance (0.01)
        # Negative Resonance (-1.0) -> High Distance (2.0)
        # Avoid division by zero
        distance = max(0.01, 1.0 - res)

        # 3. Force Calculation
        force = (m1 * m2) / (distance ** 2)

        # Cap excessive gravity (Black Holes)
        return min(force, 100.0)

    @staticmethod
    def find_resonant_cluster(center_vec: SovereignVector, radius: float = 0.3, limit: int = 10) -> List[Tuple[str, SovereignVector, float]]:
        """
        [PHASE 90] Concept Cloud Retrieval.
        Retrieves a cluster of concepts that resonate with the center vector.

        Args:
            center_vec: The gravitational center (Intent).
            radius: The resonance tolerance (0.0 = exact match, 1.0 = everything).
                    We convert this to min_resonance = 1.0 - radius.
            limit: Maximum number of concepts to retrieve.

        Returns:
            List of (Name, Vector, ResonanceScore) sorted by Resonance.
        """
        cluster = []
        min_resonance = 1.0 - radius

        # 1. Scan Roots (Heavy Mass)
        for name, data in LogosBridge.CONCEPT_MAP.items():
            vec = data["vector"]
            # Use Signed Resonance
            res = SovereignMath.signed_resonance(vec, center_vec)
            if isinstance(res, complex): res = res.real

            if res >= min_resonance:
                cluster.append((name, vec, res))

        # 2. Scan Learned (Lighter Mass)
        for name, data in LogosBridge.LEARNED_MAP.items():
            vec = data["vector"]
            res = SovereignMath.signed_resonance(vec, center_vec)
            if isinstance(res, complex): res = res.real

            if res >= min_resonance:
                cluster.append((name, vec, res))

        # 3. Sort by Resonance (Gravity)
        cluster.sort(key=lambda x: x[2], reverse=True)

        return cluster[:limit]

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
        import torch
        if hasattr(principle_vector, 'data') and isinstance(principle_vector.data, torch.Tensor):
            data_list = principle_vector.data.real.tolist()
        else:
            data_list = [v.real if hasattr(v, 'real') else v for v in principle_vector]
        trits = [1 if v > 0.5 else (-1 if v < -0.5 else 0) for v in data_list]
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
    from Core.System.somatic_logger import SomaticLogger
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
        LogosBridge.logger.sensation(f"Crystallized Concept: '{u_name}' mapped to ND.", intensity=0.6)
        
        # [PHASE 160/260] Persist to Akashic Journal (Append-only O(1))
        LogosBridge._append_to_journal(u_name, vector, description)
        
        # [PHASE 170] Carve into Cognitive Terrain (The River Carves the Land)
        LogosBridge._erode_terrain(u_name, vector)
        
        # [PHASE 260] REFRESH GPU SPECTRUM
        LogosBridge.polymerize_spectrum()

    @staticmethod
    def inject_prismatic_concept(name: str, vector: SovereignVector, roots: Dict[str, Any] = None):
        """
        [PHASE 4.5 & 5] THE ORGANIC LOOP
        Directly injects a concept into the Lexical Prism (lexical_spectrum.json).
        This closes the loop: Learned Concept -> Speakable Word.
        
        Args:
            name: Word to learn (e.g., "ZARA")
            vector: ND Meaning Vector
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
        
        # Fixed path for Linux environment
        # From Core/Cognition/logos_bridge.py, we need 3 steps to reach root
        spectrum_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "knowledge", "lexical_spectrum.json"
        )
        
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
            
            # [PHASE 260] REFRESH GPU SPECTRUM
            LogosBridge.polymerize_spectrum()
                
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
            # Map ND vector to 2D terrain coordinates using first 2 dimensions
            # Normalize to terrain grid size
            real_data = [v.real if isinstance(v, complex) else v for v in vector.data[:2]]
            x = int((real_data[0] + 1) / 2 * (terrain.resolution - 1))  # [-1,1] -> [0, res-1]
            y = int((real_data[1] + 1) / 2 * (terrain.resolution - 1))
            x = max(0, min(x, terrain.resolution - 1))
            y = max(0, min(y, terrain.resolution - 1))
            
            # Inject as a prime keyword (creates an attractor/valley)
            magnitude = float(sum(abs(v.real if isinstance(v, complex) else v) for v in vector.data) / float(len(v)))
            terrain.inject_prime_keyword(x, y, concept_name, magnitude=magnitude * 0.5)
            
            LogosBridge.logger.mechanism(f"Carved '{concept_name}' at ({x}, {y}) with depth {magnitude:.2f}")
        except Exception as e:
            LogosBridge.logger.admonition(f"Erosion failed: {e}")

    @staticmethod
    def _append_to_journal(name: str, vector: SovereignVector, description: str):
        """[PHASE 260] O(1) Atomic append to Akashic Journal."""
        try:
            raw_data = vector.data if hasattr(vector, 'data') else list(vector)
            clean_data = [float(v.real) if isinstance(v, complex) else float(v) for v in raw_data]
            entry = {
                "name": name,
                "vector": clean_data,
                "description": description,
                "stratum": MemoryStratum.LEAF.value,
                "timestamp": datetime.now().isoformat()
            }
            os.makedirs(os.path.dirname(LogosBridge.AKASHIC_JOURNAL_PATH), exist_ok=True)
            with open(LogosBridge.AKASHIC_JOURNAL_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            LogosBridge.logger.mechanism(f"Journaled '{name}' to Akashic Line.")
        except Exception as e:
            LogosBridge.logger.admonition(f"Journaling failed: {e}")

    @staticmethod
    def _persist_to_akashic():
        """[PHASE 160] Consolidate journal into main Akashic Records."""
        # This is now a "Checkpoint" operation rather than a constant one.
        try:
            data = {}
            # We assume LEARNED_MAP already contains everything (from load + learning)
            for name, concept in LogosBridge.LEARNED_MAP.items():
                vec = concept["vector"]
                raw_data = vec.data if hasattr(vec, 'data') else list(vec)
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
            # Clear journal after consolidation to prevent duplication
            if os.path.exists(LogosBridge.AKASHIC_JOURNAL_PATH):
                os.remove(LogosBridge.AKASHIC_JOURNAL_PATH)
            LogosBridge.logger.mechanism(f"Knowledge Consolidated: {len(data)} concepts in main record.")
        except Exception as e:
            LogosBridge.logger.admonition(f"Consolidation failed: {e}")

    @staticmethod
    def _load_from_akashic():
        """[PHASE 160/260] Restore from Base Record + Journal."""
        try:
            # 1. Load Base
            if os.path.exists(LogosBridge.AKASHIC_PATH):
                with open(LogosBridge.AKASHIC_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for name, concept in data.items():
                    LogosBridge.LEARNED_MAP[name] = {
                        "vector": SovereignVector(concept["vector"]),
                        "description": concept.get("description", ""),
                        "stratum": MemoryStratum(concept.get("stratum", 0))
                    }
            
            # 2. Load Journal (Deltas)
            journal_count = 0
            if os.path.exists(LogosBridge.AKASHIC_JOURNAL_PATH):
                with open(LogosBridge.AKASHIC_JOURNAL_PATH, 'r', encoding='utf-8') as f:
                    for line in f:
                        entry = json.loads(line)
                        LogosBridge.LEARNED_MAP[entry["name"]] = {
                            "vector": SovereignVector(entry["vector"]),
                            "description": entry["description"],
                            "stratum": MemoryStratum(entry["stratum"])
                        }
                        journal_count += 1
            
            LogosBridge.logger.sensation(f"Akashic Awakened: {len(LogosBridge.LEARNED_MAP)} concepts (+{journal_count} journaled).", intensity=0.7)
        except Exception as e:
            LogosBridge.logger.admonition(f"Restoration failed: {e}")

    @staticmethod
    def identify_concept(principle_vector: SovereignVector) -> Tuple[str, float]:
        """[Legacy] Wraps find_closest_concept for backward compatibility."""
        return LogosBridge.find_closest_concept(principle_vector)

    @staticmethod
    def find_closest_concept(principle_vector: SovereignVector) -> Tuple[str, float]:
        """
        [PHASE 260] Vectorized Universal Search.
        Uses XOR bitwise comparison for rapid candidate filtering,
        followed by high-fidelity rotor/mass mapping.
        """
        if not LogosBridge._CONCEPT_BITMASKS:
            LogosBridge._build_bitmask_cache()

        # 1. Transcribe query vector to DNA and get bitmasks
        dna_in = LogosBridge.transcribe_to_dna(principle_vector)
        pos_in, neg_in = LogosBridge.dna_to_masks(dna_in)

        # 2. Turing XOR Comparison (Rapid Candidate Selection)
        candidates = []
        for name, (cached_pos, cached_neg) in LogosBridge._CONCEPT_BITMASKS.items():
            diff_pos = pos_in ^ cached_pos
            diff_neg = neg_in ^ cached_neg
            diff_count = bin(diff_pos).count('1') + bin(diff_neg).count('1')
            # Trinary Match Score (Hamming-based)
            xor_match = (21 - diff_count) / 21.0
            candidates.append((name, xor_match))

        # Sort by XOR match score and take top K (e.g. 5)
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [c[0] for c in candidates[:5]]

        if not top_candidates:
            return "UNKNOWN/CHAOS", 0.0

        # 3. High-Fidelity Scoring on Top Candidates (Hybrid Resonance)
        best_concept = top_candidates[0]
        max_resonance = -2.0
        v_norm = principle_vector.normalize()

        for name in top_candidates:
            target = LogosBridge.recall_concept_vector(name).normalize()
            
            # Complex Euler interaction (i * i = -1) - resonance calculation
            resonance = SovereignMath.resonance(v_norm, target)
            if isinstance(resonance, complex): 
                resonance = resonance.real
                
            mass = LogosBridge.get_stratum_mass(name)
            weighted_resonance = resonance * mass
            
            # Blend XOR score and vector resonance
            xor_weight = next(c[1] for c in candidates if c[0] == name)
            combined_score = (weighted_resonance * 0.7) + (xor_weight * 0.3 * mass)

            if combined_score > max_resonance:
                max_resonance = combined_score
                best_concept = name

        return best_concept, float(max_resonance)

    @staticmethod
    def _sequential_find_closest(principle_vector: SovereignVector) -> Tuple[str, float]:
        """Legacy sequential search CPU fallback, routed to the optimized find_closest_concept."""
        return LogosBridge.find_closest_concept(principle_vector)

    @staticmethod
    def discover_novel_vibration(vec: SovereignVector, threshold: float = 0.5) -> bool:
        """
        [PHASE 74] Identifies if a ND vibration is 'Unknown' to the manifold.
        Normalized threshold (0-1).
        """
        best_concept, best_score = LogosBridge.find_closest_concept(vec)
        # We need to de-scale the score by the stratum mass to get raw dot product
        # Roots have mass 3. So score / 3.0
        mass = LogosBridge.get_stratum_mass(best_concept)
        raw_dot = best_score / mass
        
        return raw_dot < threshold

    @staticmethod
    def suggest_proto_logos(vec: SovereignVector) -> str:
        """
        [PHASE 74] Generates a temporary Seed ID for a novel vibration.
        """
        import uuid
        seed_id = str(uuid.uuid4())[:4].upper()
        return f"PROTO_{seed_id}"

    @staticmethod
    def calculate_text_resonance(text: str) -> SovereignVector:
        # [PHASE 130] Unified Inhalation
        # [V2.0] Now returns a tuple (vector, residue) if possible, but keep signature for compat.
        # Ideally, we attach residue to the vector or return it separately.
        # For now, inhale_text handles the logic.
        return LogosBridge.inhale_text(text)

    @staticmethod
    def calculate_analog_residue(text: str, vector: SovereignVector) -> float:
        """
        [DOCTRINE OF THE PRISM]
        Calculates the 'Quantization Loss' (Residue) when converting Analog Text to Digital Vector.

        Residue = (Analog Complexity) - (Digital Certainty)

        - Analog Complexity: Richness of adjectives, length, metaphors (entropy).
        - Digital Certainty: Magnitude of the resulting vector (how well it mapped to known concepts).
        """
        import math

        # 1. Measure Analog Complexity (Entropy estimate)
        # Length, unique words, emotional words
        words = text.split()
        unique_words = set(words)

        # Base entropy: longer, more unique text is harder to compress
        analog_complexity = math.log(len(words) + 1) + (len(unique_words) / (len(words) + 1))

        # 2. Measure Digital Certainty (Resonance Mass)
        # High norm means we found strong existing concepts to map to.
        # Low norm means we just averaged a bunch of noise (high loss).
        digital_certainty = 0.0
        if hasattr(vector, 'norm'):
            norm_val = vector.norm()
            if isinstance(norm_val, complex): norm_val = norm_val.real
            # A 'perfect' match usually has norm ~1.0 or higher if multiple concepts align
            digital_certainty = float(norm_val)

        # 3. Calculate Residue
        # If complexity is high but certainty is low, Residue is HIGH (We missed the meaning).
        # If complexity is low and certainty is high, Residue is LOW (We got it).

        # [V2.0 FIX] Ensure residue is not zero-clamped too early if norms are small (like in mocks)
        # If digital_certainty is 0.0 (Mock), Residue = Analog Complexity

        residue = max(0.0, analog_complexity - digital_certainty)

        # Scale for usability (0.0 - 1.0 range preferred)
        # Increase gain to ensure visibility
        residue = min(1.0, residue * 0.5)

        return residue

    @staticmethod
    def inhale_text(text: str) -> SovereignVector:
        """
        [PHASE 130] Digestion and Crystallization.
        Decomposes text, identifies novel concepts, and crystallizes them.
        [V2.0] Calculates Analog Residue and attaches it to the vector.
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
            
        final_vector = consensus_vector.normalize()

        # [V2.0] Calculate Residue
        residue = LogosBridge.calculate_analog_residue(text, final_vector)

        # Attach residue to the vector object (Duck Typing)
        try:
            final_vector.analog_residue = residue
        except AttributeError:
            # Fallback if SovereignVector uses __slots__ or is immutable
            # We return a wrapper or just the vector (residue loss is acceptable in fallback)
            pass

        return final_vector

    @staticmethod
    def parse_narrative_to_torque(text: str) -> Any:
        """
        [PHASE 72] Internal Echo.
        Converts narrative text back into a 4D torque vector for the manifold.
        """
        # 1. Get ND Semantic Vector
        vec = LogosBridge.calculate_text_resonance(text)
        
        # 2. Project ND -> 4D (Standard Projection)
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
        [PHASE 90] Converts a ND Semantic Vector into a 4D Engine Torque.
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
