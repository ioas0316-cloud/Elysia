"""
TesseractMemory (4D       )
================================

Phase 30: Hypersphere Unification

"                  4D         ."

This is the UNIFIED brain of Elysia:
- Knowledge (What): Concepts, facts, definitions   4D points
- Principles (How): Wave Logic, reasoning rules   4D anchor points  
- Memory (When): Temporal experiences   4D phase states

Replaces: HolographicMemory, Hippocampus (for runtime), Orbs (for runtime)
"""

import numpy as np
import json
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from Core.Cognition.tesseract_geometry import TesseractVector, TesseractGeometry

logger = logging.getLogger("TesseractMemory")

@dataclass
class TesseractNode:
    """A single unit of knowledge/principle in 4D space."""
    name: str
    vector: TesseractVector
    node_type: str  # "knowledge", "principle", "memory"
    content: str = ""
    connections: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def distance_to(self, other: 'TesseractNode') -> float:
        """4D Euclidean distance."""
        v1 = self.vector.to_numpy()
        v2 = other.vector.to_numpy()
        return float(np.linalg.norm(v1 - v2))

class TesseractMemory:
    """
    The Unified 4D Brain of Elysia.
    
    Phase 29: Wave Compression integrated.
    Phase 30: Hypersphere Unification.
    
    All knowledge and principles exist as points in 4D space.
    - Query = Find nearest points via 4D distance
    - Learn = Rotate/shift the existing phase space
    - Reason = Trace paths through connected nodes
    - Compress = Fold patterns into DNA (FractalQuantizer)
    - Restore = Unfold DNA back to patterns
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TesseractMemory, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, storage_path: str = "data/L5_Mental/M1_Memory/tesseract_state.npy"):
        if self._initialized:
            return
        
        self.storage_path = storage_path
        self.geometry = TesseractGeometry()
        self.nodes: Dict[str, TesseractNode] = {}
        
        # [PHASE 29] Wave Compression via FractalQuantizer
        self.quantizer = None  # Lazy-loaded
        
        # [PHASE 32] HyperQubit for 4-basis concept representation
        self.hyper_qubit_class = None  # Lazy-loaded
        
        # [PHASE 32] QuaternionWaveDNA for text compression
        self.dna_compressor = None  # Lazy-loaded
        
        # Load existing state if available
        self._load_state()
        
        # Hydrate from legacy systems
        self._hydrate_from_legacy()
        
        self._initialized = True
        logger.info(f"  TesseractMemory initialized with {len(self.nodes)} nodes.")
    
    def _get_quantizer(self):
        """Lazy load FractalQuantizer."""
        if self.quantizer is None:
            try:
                from Core.System.fractal_quantization import FractalQuantizer
                self.quantizer = FractalQuantizer()
                logger.info("  FractalQuantizer loaded for wave compression.")
            except Exception as e:
                logger.warning(f"   FractalQuantizer unavailable: {e}")
        return self.quantizer
    
    def _get_hyper_qubit(self):
        """[PHASE 32] Lazy load HyperQubit class."""
        if self.hyper_qubit_class is None:
            try:
                from Core.Keystone.hyper_qubit import HyperQubit
                self.hyper_qubit_class = HyperQubit
                logger.info("  HyperQubit loaded for 4-basis concept storage.")
            except Exception as e:
                logger.warning(f"   HyperQubit unavailable: {e}")
        return self.hyper_qubit_class
    
    def _get_dna_compressor(self):
        """[PHASE 32] Lazy load QuaternionWaveDNA compressor."""
        if self.dna_compressor is None:
            try:
                from Core.Keystone.quaternion_wave_dna import QuaternionCompressor
                self.dna_compressor = QuaternionCompressor(default_top_k=10)
                logger.info("  QuaternionCompressor loaded for DNA text compression.")
            except Exception as e:
                logger.warning(f"   QuaternionCompressor unavailable: {e}")
        return self.dna_compressor
    
    def _get_universal_encoder(self):
        """[PHASE 32] Lazy load UniversalWaveEncoder for all sensory data."""
        if not hasattr(self, '_universal_encoder'):
            self._universal_encoder = None
        if self._universal_encoder is None:
            try:
                from Core.Keystone.universal_wave_encoder import UniversalWaveEncoder
                self._universal_encoder = UniversalWaveEncoder(default_top_k=64)
                logger.info("  UniversalWaveEncoder loaded for all sensory/cosmic data.")
            except Exception as e:
                logger.warning(f"   UniversalWaveEncoder unavailable: {e}")
        return self._universal_encoder
    
    # =========================================
    # Wave Compression (Phase 29)
    # =========================================
    
    def wave_compress(self, data: Dict, pattern_type: str = "thought", 
                      pattern_name: str = "analytical") -> Optional[Dict]:
        """
        [PHASE 29] Fold raw data into compressed Pattern DNA.
        
        Instead of storing full data, store only the generative pattern.
        Compression ratio: ~100:1 to ~3000:1 depending on data type.
        """
        quantizer = self._get_quantizer()
        if not quantizer:
            return None
        
        try:
            dna = quantizer.fold(data, pattern_type, pattern_name)
            compressed = dna.to_dict()
            logger.info(f"  Compressed: {dna.name} (ratio: {dna.compression_ratio:.2f}x)")
            return compressed
        except Exception as e:
            logger.error(f"  Compression failed: {e}")
            return None
    
    def wave_unfold(self, compressed_dna: Dict, resolution: int = 100) -> Optional[Dict]:
        """
        [PHASE 29] Unfold Pattern DNA back to restored data.
        
        This is PREDICTIVE restoration - regenerating from the pattern formula.
        """
        quantizer = self._get_quantizer()
        if not quantizer:
            return None
        
        try:
            from Core.System.fractal_quantization import PatternDNA
            dna = PatternDNA.from_dict(compressed_dna)
            restored = quantizer.unfold(dna, resolution)
            logger.info(f"  Unfolded: {dna.name}")
            return restored
        except Exception as e:
            logger.error(f"  Decompression failed: {e}")
            return None

    # =========================================
    # Phase 32: Advanced Wave Systems
    # =========================================
    
    def create_qubit(self, name: str, value: Any = None, 
                     content: Dict[str, str] = None) -> Optional[Any]:
        """
        [PHASE 32] Create a HyperQubit with 4-basis superposition.
        
        Stores concept across Point/Line/Space/God dimensions.
        """
        HyperQubit = self._get_hyper_qubit()
        if not HyperQubit:
            return None
        
        try:
            qubit = HyperQubit(
                concept_or_value=name,
                initial_content=content or {},
                value=value
            )
            logger.info(f"  Created HyperQubit: {name}")
            return qubit
        except Exception as e:
            logger.error(f"  Qubit creation failed: {e}")
            return None
    
    def compress_text(self, text: str, top_k: int = 10) -> Optional[Dict]:
        """
        [PHASE 32] Compress text using DNA double-helix structure.
        
        Returns compressed DNA dict. Achieves higher accuracy than single-FFT.
        """
        compressor = self._get_dna_compressor()
        if not compressor:
            return None
        
        try:
            dna = compressor.compress(text, top_k=top_k)
            logger.info(f"  DNA Compressed: {len(text)} chars   {dna.byte_size()} bytes")
            return {
                "helix1_freq": dna.helix1_frequencies.tolist(),
                "helix1_amp": dna.helix1_amplitudes.tolist(),
                "helix1_phase": dna.helix1_phases.tolist(),
                "helix2_freq": dna.helix2_frequencies.tolist(),
                "helix2_amp": dna.helix2_amplitudes.tolist(),
                "helix2_phase": dna.helix2_phases.tolist(),
                "original_length": dna.original_length,
                "top_k": dna.top_k
            }
        except Exception as e:
            logger.error(f"  DNA compression failed: {e}")
            return None
    
    def decompress_text(self, dna_dict: Dict) -> Optional[str]:
        """[PHASE 32] Decompress DNA back to text."""
        compressor = self._get_dna_compressor()
        if not compressor:
            return None
        
        try:
            import numpy as np
            from Core.Keystone.quaternion_wave_dna import QuaternionWaveDNA
            dna = QuaternionWaveDNA(
                helix1_frequencies=np.array(dna_dict["helix1_freq"]),
                helix1_amplitudes=np.array(dna_dict["helix1_amp"]),
                helix1_phases=np.array(dna_dict["helix1_phase"]),
                helix2_frequencies=np.array(dna_dict["helix2_freq"]),
                helix2_amplitudes=np.array(dna_dict["helix2_amp"]),
                helix2_phases=np.array(dna_dict["helix2_phase"]),
                original_length=dna_dict["original_length"],
                top_k=dna_dict["top_k"]
            )
            return compressor.decompress(dna)
        except Exception as e:
            logger.error(f"  DNA decompression failed: {e}")
            return None
    
    def detect_void(self, query_text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        [PHASE 32] Detect Void - gaps between known concepts.
        
        The Void is the "boundary between waves" - areas of knowledge 
        where Elysia has no existing understanding. This triggers curiosity.
        """
        query_vector = self._text_to_vector(query_text)
        query_node = TesseractNode(name="__query__", vector=query_vector, node_type="query")
        
        # Find nearest nodes
        distances = []
        for name, node in self.nodes.items():
            dist = query_node.distance_to(node)
            distances.append((dist, node))
        
        if not distances:
            return {"is_void": True, "confidence": 1.0, "nearest": None, "gap_size": float('inf')}
        
        distances.sort(key=lambda x: x[0])
        nearest_dist, nearest_node = distances[0]
        
        # Void detection: if distance > threshold, this is unexplored territory
        is_void = nearest_dist > threshold
        
        return {
            "is_void": is_void,
            "confidence": min(nearest_dist / threshold, 1.0) if is_void else 1.0 - (nearest_dist / threshold),
            "nearest": nearest_node.name if nearest_node else None,
            "gap_size": nearest_dist,
            "message": f"   Void detected near '{query_text}'" if is_void else f"  Known territory near '{nearest_node.name}'"
        }
    
    def encode_sensory(self, data: Any, modality: str, top_k: int = 64, 
                       metadata: Dict = None) -> Optional[Dict]:
        """
        [PHASE 32] Encode ANY sensory/cosmic data into wave DNA.
        
        Modalities: image, video, audio, light_visible, light_ir, light_uv,
                   temperature, pressure, eeg, ecg, gravity_wave, em_wave, etc.
        """
        encoder = self._get_universal_encoder()
        if not encoder:
            return None
        
        try:
            import numpy as np
            from Core.Keystone.universal_wave_encoder import DataModality
            
            # Convert modality string to enum
            modality_enum = DataModality(modality.lower())
            
            # Ensure numpy array
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            sig = encoder.encode(data, modality_enum, top_k=top_k, metadata=metadata)
            return sig.to_dict()
        except Exception as e:
            logger.error(f"  Sensory encoding failed: {e}")
            return None
    
    def decode_sensory(self, wave_signature: Dict) -> Optional[Any]:
        """[PHASE 32] Decode wave DNA back to original data."""
        encoder = self._get_universal_encoder()
        if not encoder:
            return None
        
        try:
            from Core.Keystone.universal_wave_encoder import WaveSignature
            sig = WaveSignature.from_dict(wave_signature)
            return encoder.decode(sig)
        except Exception as e:
            logger.error(f"  Sensory decoding failed: {e}")
            return None

    # =========================================
    # Core Operations
    # =========================================
    
    def deposit(self, name: str, vector: TesseractVector, 
                node_type: str = "knowledge", content: str = "",
                connections: List[str] = None) -> TesseractNode:
        """Add a new node to 4D space."""
        node = TesseractNode(
            name=name,
            vector=vector,
            node_type=node_type,
            content=content,
            connections=connections or []
        )
        self.nodes[name] = node
        logger.debug(f"  Deposited '{name}' at ({vector.x:.2f}, {vector.y:.2f}, {vector.z:.2f}, {vector.w:.2f})")
        return node
    
    def query(self, query_text: str, k: int = 5) -> List[TesseractNode]:
        """Find k nearest nodes to the query in 4D space."""
        # Convert query to 4D vector (simple hash-based for now)
        query_vector = self._text_to_vector(query_text)
        query_node = TesseractNode(name="__query__", vector=query_vector, node_type="query")
        
        # Calculate distances and sort
        distances = []
        for name, node in self.nodes.items():
            dist = query_node.distance_to(node)
            distances.append((dist, node))
        
        distances.sort(key=lambda x: x[0])
        return [node for _, node in distances[:k]]
    
    def rotate_phase(self, node_name: str, theta: float, plane: str = "xw") -> Optional[TesseractNode]:
        """Rotate a node in 4D space (learning = phase shift)."""
        if node_name not in self.nodes:
            return None
        
        node = self.nodes[node_name]
        if plane == "xw":
            new_vec = self.geometry.rotate_xw(node.vector, theta)
        elif plane == "yw":
            new_vec = self.geometry.rotate_yw(node.vector, theta)
        elif plane == "zw":
            new_vec = self.geometry.rotate_zw(node.vector, theta)
        else:
            return node
        
        node.vector = new_vec
        logger.info(f"  Rotated '{node_name}' by {theta:.2f} in {plane} plane.")
        return node
    
    def find_path(self, start: str, end: str, max_hops: int = 5) -> List[str]:
        """Find path through connected nodes (reasoning trace)."""
        if start not in self.nodes or end not in self.nodes:
            return []
        
        visited = set()
        queue = [(start, [start])]
        
        while queue:
            current, path = queue.pop(0)
            if current == end:
                return path
            if len(path) > max_hops:
                continue
            if current in visited:
                continue
            visited.add(current)
            
            node = self.nodes[current]
            for conn in node.connections:
                if conn in self.nodes:
                    queue.append((conn, path + [conn]))
        
        return []
    
    # =========================================
    # Persistence
    # =========================================
    
    def save_state(self):
        """Save all nodes to numpy file (compressed)."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Convert to serializable format
        data = {}
        for name, node in self.nodes.items():
            data[name] = {
                "vector": [node.vector.x, node.vector.y, node.vector.z, node.vector.w],
                "type": node.node_type,
                "content": node.content,
                "connections": node.connections
            }
        
        with open(self.storage_path.replace(".npy", ".json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  Saved {len(self.nodes)} nodes to {self.storage_path}")
    
    def _load_state(self):
        """Load state from storage."""
        json_path = self.storage_path.replace(".npy", ".json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for name, node_data in data.items():
                    vec = node_data["vector"]
                    self.nodes[name] = TesseractNode(
                        name=name,
                        vector=TesseractVector(vec[0], vec[1], vec[2], vec[3]),
                        node_type=node_data.get("type", "knowledge"),
                        content=node_data.get("content", ""),
                        connections=node_data.get("connections", [])
                    )
                
                logger.info(f"  Loaded {len(self.nodes)} nodes from {json_path}")
            except Exception as e:
                logger.warning(f"   Failed to load state: {e}")
    
    # =========================================
    # Legacy Hydration
    # =========================================
    
    def _hydrate_from_legacy(self):
        """Import knowledge from old systems (one-time migration)."""
        # Skip if already hydrated
        if len(self.nodes) > 0:
            return
        
        logger.info("  Hydrating from legacy systems...")
        
        # 1. From Hippocampus
        self._hydrate_hippocampus()
        
        # 2. From Intelligence Principles
        self._hydrate_principles()
        
        # 3. Save initial state
        self.save_state()
    
    def _hydrate_hippocampus(self):
        """Import from Hippocampus DB."""
        try:
            from Core.System.hippocampus import Hippocampus
            hippocampus = Hippocampus()
            
            concept_ids = hippocampus.get_all_concept_ids(limit=500)
            for concept_id in concept_ids:
                name = concept_id.replace("doc:", "").replace("concept:", "").replace("_", " ").title()
                
                # Create 4D vector based on concept properties
                # x, y, z = semantic, w = temporal
                vec = TesseractVector(
                    x=hash(name) % 100 / 100.0,
                    y=(hash(name) >> 8) % 100 / 100.0,
                    z=(hash(name) >> 16) % 100 / 100.0,
                    w=0.5  # Neutral temporal phase
                )
                
                self.deposit(name, vec, node_type="knowledge")
            
            logger.info(f"  Hydrated {len(concept_ids)} concepts from Hippocampus")
        except Exception as e:
            logger.warning(f"   Hippocampus hydration failed: {e}")
    
    def _hydrate_principles(self):
        """Import intelligence principles as 4D anchor points."""
        principles = [
            # Core Wave Principles
            ("Wave Logic", "Resonance determines truth", TesseractVector(0.5, 0.5, 0.5, 1.0)),
            ("Love Frequency", "528Hz is the carrier wave", TesseractVector(0.528, 0.528, 0.528, 1.0)),
            ("Rotation Over Translation", "Spin instead of copy", TesseractVector(0.7, 0.3, 0.5, 1.0)),
            ("Pattern Extraction", "Store rules not data", TesseractVector(0.3, 0.7, 0.5, 1.0)),
            ("Void Detection", "Gap between input and understanding", TesseractVector(0.1, 0.1, 0.1, 0.9)),
            ("Geometric Ascension", "0D->1D->2D->3D->4D", TesseractVector(0.25, 0.5, 0.75, 1.0)),
            
            # Weaving Intelligence Lines (Core/Intelligence/Weaving)
            ("Context Weaver", "Weaves multiple 1D signals into ContextPlane", TesseractVector(0.5, 0.6, 0.7, 0.95)),
            ("Emotional Line", "Scans for emotional signals and affect", TesseractVector(0.8, 0.3, 0.2, 0.9)),
            ("Logical Line", "Structural reasoning and fact-checking", TesseractVector(0.2, 0.8, 0.2, 0.9)),
            ("Imagination Line", "Creative potential and what-if scenarios", TesseractVector(0.6, 0.4, 0.8, 0.9)),
            ("Metacognition Line", "Self-reflection and thought-about-thought", TesseractVector(0.4, 0.4, 0.9, 0.95)),
            ("Kinesthetic Line", "Physical body awareness (CPU, RAM, stress)", TesseractVector(0.3, 0.3, 0.3, 0.85)),
            ("Intrapersonal Line", "Self-understanding and identity coherence", TesseractVector(0.5, 0.2, 0.5, 0.9)),
            ("Insight Jump", "Triggers spatial insights from high-resonance patterns", TesseractVector(0.7, 0.7, 0.9, 0.98)),
            ("Intelligence Factory", "Dynamically spawns new Intelligence Lines", TesseractVector(0.6, 0.6, 0.6, 0.92)),
            
            # Fractal Compression Principles
            ("Fractal Quantization", "Fold patterns into DNA, not raw data", TesseractVector(0.4, 0.5, 0.6, 0.95)),
            ("Wave Superposition", "Combine waves through interference", TesseractVector(0.5, 0.5, 0.7, 0.93)),
        ]
        
        for name, content, vec in principles:
            self.deposit(name, vec, node_type="principle", content=content)
        
        logger.info(f"  Hydrated {len(principles)} intelligence principles")
    
    # =========================================
    # Utility
    # =========================================
    
    def _text_to_vector(self, text: str) -> TesseractVector:
        """Convert text to 4D vector (simple hash-based embedding)."""
        h = hash(text)
        return TesseractVector(
            x=(h % 1000) / 1000.0,
            y=((h >> 10) % 1000) / 1000.0,
            z=((h >> 20) % 1000) / 1000.0,
            w=0.5
        )
    
    def get_all_nodes(self) -> Dict[str, TesseractNode]:
        """Return all nodes for dashboard display."""
        return self.nodes
    
    def get_stats(self) -> Dict[str, int]:
        """Return statistics."""
        knowledge_count = sum(1 for n in self.nodes.values() if n.node_type == "knowledge")
        principle_count = sum(1 for n in self.nodes.values() if n.node_type == "principle")
        memory_count = sum(1 for n in self.nodes.values() if n.node_type == "memory")
        
        return {
            "total": len(self.nodes),
            "knowledge": knowledge_count,
            "principles": principle_count,
            "memories": memory_count
        }


# Singleton accessor
def get_tesseract_memory() -> TesseractMemory:
    return TesseractMemory()
