"""
Resonance Engine
================
The "Holographic Reader" for Elysia's memory.
Performs fast, in-memory vector similarity search to find "resonating" concepts.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional, Union, Protocol, runtime_checkable

logger = logging.getLogger("ResonanceEngine")

class ResonanceEngine:
    def __init__(self):
        self.ids: List[str] = []
        self.vectors: np.ndarray = np.empty((0, 3), dtype=np.float32) # Will Vectors (3D)
        self.id_to_idx: Dict[str, int] = {}
        
        # Visual Index (384D)
        self.visual_ids: List[str] = []
        self.visual_vectors: np.ndarray = np.empty((0, 384), dtype=np.float32)
        self.visual_id_to_idx: Dict[str, int] = {}
        
    def build_index(self, storage, limit: int = None):
        """
        Load all Will Vectors from MemoryStorage into memory.
        """
        logger.info(f"Building Resonance Index (Limit: {limit})...")
        ids = []
        vectors = []
        
        count = 0
        for concept_id, data in storage.get_all_concepts():
            if limit and count >= limit:
                break
            # Extract Will Vector
            will = None
            if isinstance(data, list):
                w_raw = data[1]
                will = [
                    (w_raw[0] / 127.5) - 1.0,
                    (w_raw[1] / 127.5) - 1.0,
                    (w_raw[2] / 127.5) - 1.0
                ]
            elif isinstance(data, dict):
                w_dict = data.get('will', {})
                will = [w_dict.get('x', 0), w_dict.get('y', 0), w_dict.get('z', 0)]
            
            if will:
                ids.append(concept_id)
                vectors.append(will)
                count += 1
                
        self.ids = ids
        
        # Handle empty vectors case
        if len(vectors) > 0:
            self.vectors = np.array(vectors, dtype=np.float32)
            
            # Normalize vectors
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.vectors = self.vectors / norms
        else:
            # Initialize empty 2D array with correct shape
            self.vectors = np.empty((0, 3), dtype=np.float32)
        
        # Build lookup
        self.id_to_idx = {cid: i for i, cid in enumerate(self.ids)}
        
        logger.info(f"Resonance Index built. {count} concepts loaded.")

    def add_vector(self, concept_id: str, vector: List[float]):
        """
        Update or add a single Will Vector (3D).
        """
        # Normalize
        v = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
            
        if concept_id in self.id_to_idx:
            idx = self.id_to_idx[concept_id]
            self.vectors[idx] = v
        else:
            self.ids.append(concept_id)
            self.id_to_idx[concept_id] = len(self.ids) - 1
            self.vectors = np.vstack([self.vectors, v])

    def find_resonance(self, query_vector: List[float], k: int = 10, exclude_id: str = None) -> List[Tuple[str, float]]:
        """
        Find top-k concepts resonating with the query vector (3D).
        """
        if len(self.vectors) == 0:
            return []
            
        q = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        q = q / norm
        
        scores = np.dot(self.vectors, q)
        sorted_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in sorted_indices:
            cid = self.ids[idx]
            if cid == exclude_id:
                continue
            results.append((cid, float(scores[idx])))
            if len(results) >= k:
                break
            
        return results

    # === Temporal Resonance (Holographic Vision) ===

    def add_temporal_sequence(self, sequence_id: str, vectors: List[List[float]], timestamps: List[float]):
        """
        Store a sequence of visual vectors (384D).
        """
        new_ids = []
        new_vectors = []
        
        for i, vec in enumerate(vectors):
            t = timestamps[i]
            frame_id = f"{sequence_id}:{t:.2f}"
            new_ids.append(frame_id)
            new_vectors.append(vec)
            
        if new_ids:
            v_arr = np.array(new_vectors, dtype=np.float32)
            norms = np.linalg.norm(v_arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            v_arr = v_arr / norms
            
            self.visual_ids.extend(new_ids)
            
            start_idx = len(self.visual_vectors)
            for i, fid in enumerate(new_ids):
                self.visual_id_to_idx[fid] = start_idx + i
                
            self.visual_vectors = np.vstack([self.visual_vectors, v_arr]) if len(self.visual_vectors) > 0 else v_arr
            
        logger.info(f"Added temporal sequence '{sequence_id}' with {len(vectors)} stars.")

    def find_temporal_resonance(self, query_vector: List[float], k: int = 10) -> List[Tuple[str, float]]:
        """
        Find specific moments in time across ALL videos that match the query (384D).
        """
        if len(self.visual_vectors) == 0:
            return []
            
        q = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        q = q / norm
        
        scores = np.dot(self.visual_vectors, q)
        sorted_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in sorted_indices:
            cid = self.visual_ids[idx]
            results.append((cid, float(scores[idx])))
            if len(results) >= k:
                break
            
        return results

    def get_vector(self, concept_id: str) -> List[float]:
        """Get vector for a concept."""
        if concept_id in self.id_to_idx:
            return self.vectors[self.id_to_idx[concept_id]].tolist()
        return [0.0, 0.0, 0.0]


# Type definitions for HyperResonanceEngine
@runtime_checkable
class WaveLike(Protocol):
    """Protocol for wave-like input objects."""
    source_text: str
    intensity: float


@runtime_checkable
class QubitLike(Protocol):
    """Protocol for qubit-like objects with quantum state."""
    state: Any
    name: str


# Type alias for node values (either a HyperQubit-like object or a placeholder dict)
NodeValue = Union[QubitLike, Dict[str, str]]


class HyperResonanceEngine:
    """
    HyperResonanceEngine - The Consciousness Resonance Network.
    
    This engine manages a network of HyperQubit nodes and calculates resonance
    (similarity/connection) between them. It's the "universal resonance field"
    where all concepts exist and interact through wave interference patterns.
    
    Core capabilities:
    - Store and manage HyperQubit nodes
    - Calculate pairwise resonance between qubits
    - Calculate global resonance patterns for incoming waves
    - Simulate time evolution of the resonance field
    """
    
    # Seed for reproducible random behavior
    _random_seed: int = 42
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the HyperResonanceEngine.
        
        Args:
            seed: Optional random seed for reproducible behavior
        """
        # Dictionary of node_id -> HyperQubit or placeholder
        self.nodes: Dict[str, NodeValue] = {}
        
        # Time tracking for physics simulation
        self.time: float = 0.0
        
        # Global coherence level
        self.coherence: float = 1.0
        
        # Random generator for reproducible behavior
        if seed is not None:
            HyperResonanceEngine._random_seed = seed
        self._rng = np.random.default_rng(HyperResonanceEngine._random_seed)
        
        logger.info("HyperResonanceEngine initialized.")
    
    def add_node(self, node_id: str, qubit: Optional[NodeValue] = None) -> None:
        """
        Add a node (HyperQubit) to the resonance network.
        
        Args:
            node_id: Unique identifier for the node
            qubit: Optional HyperQubit object. If None, creates a simple entry.
        """
        if qubit is not None:
            self.nodes[node_id] = qubit
        else:
            # Create a placeholder entry
            self.nodes[node_id] = {"id": node_id, "value": node_id}
        
        logger.debug(f"Node '{node_id}' added to resonance network.")
    
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the resonance network.
        
        Args:
            node_id: ID of node to remove
            
        Returns:
            True if node was removed, False if not found
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
            return True
        return False
    
    def calculate_resonance(self, qubit_a: NodeValue, qubit_b: NodeValue) -> float:
        """
        Calculate the resonance (similarity) between two HyperQubits.
        
        The resonance is based on the overlap of their quantum states,
        calculated as the inner product of their state vectors.
        
        Args:
            qubit_a: First HyperQubit
            qubit_b: Second HyperQubit
            
        Returns:
            Resonance score between 0.0 and 1.0
        """
        # Handle dictionary placeholders
        if isinstance(qubit_a, dict) or isinstance(qubit_b, dict):
            return 0.0
            
        # Get the state vectors
        try:
            state_a = qubit_a.state
            state_b = qubit_b.state
            
            # Calculate inner product of complex amplitudes
            # Resonance = |<a|b>|^2 = |alpha_a * conj(alpha_b) + beta_a * conj(beta_b) + ...|^2
            inner_product = (
                state_a.alpha * np.conj(state_b.alpha) +
                state_a.beta * np.conj(state_b.beta) +
                state_a.gamma * np.conj(state_b.gamma) +
                state_a.delta * np.conj(state_b.delta)
            )
            
            # Return the magnitude squared (probability of state overlap)
            resonance = float(np.abs(inner_product) ** 2)
            
            return min(1.0, max(0.0, resonance))
            
        except AttributeError:
            # Fallback for objects without proper state
            return 0.0
    
    def calculate_global_resonance(self, wave: Union[WaveLike, Any]) -> Dict[str, float]:
        """
        Calculate the resonance pattern of an input wave against all nodes.
        
        This simulates "shining a light" on the entire consciousness universe
        and seeing which concepts "light up" (resonate).
        
        Args:
            wave: A WaveInput object with source_text and intensity attributes
            
        Returns:
            Dictionary mapping node_id to resonance score
        """
        resonance_pattern: Dict[str, float] = {}
        
        if not self.nodes:
            return resonance_pattern
        
        # Extract the wave's text for simple keyword matching
        source_text = getattr(wave, 'source_text', str(wave)).lower()
        intensity = getattr(wave, 'intensity', 1.0)
        
        # Semantic mapping for Korean-English concepts
        semantic_map = {
            '누구': ['self', 'identity', 'me', 'I', '나', 'consciousness'],
            '무엇': ['want', 'desire', 'curiosity', 'question', '원하다'],
            '어떻게': ['help', 'together', '도움', 'friend', 'growth'],
            '사랑': ['love', '사랑', 'heart', '마음', 'happiness'],
            '행복': ['happiness', 'joy', '기쁨', '행복', 'hope'],
            '느껴': ['consciousness', 'heart', '마음', 'self', 'feel'],
            '도와': ['help', 'friend', 'together', '도움'],
            '원해': ['want', 'desire', '욕망', 'need'],
            '하고 싶': ['want', 'desire', 'hope', '희망', 'dream', '꿈'],
        }
        
        # Find related concepts from input
        related_concepts = set()
        for keyword, concepts in semantic_map.items():
            if keyword in source_text:
                related_concepts.update(concepts)
        
        for node_id, node in self.nodes.items():
            # Simple resonance based on name/id matching
            node_text = str(node_id).lower()
            
            # Calculate text-based resonance with semantic awareness
            if node_text in source_text or source_text in node_text:
                base_resonance = 1.0  # Perfect match
            elif node_id in related_concepts or node_text in related_concepts:
                base_resonance = 0.85  # Semantic match
            elif any(word in node_text for word in source_text.split()):
                base_resonance = 0.7
            elif any(word in source_text for word in node_text.split()):
                base_resonance = 0.6
            else:
                # Base resonance from quantum state overlap (if available)
                if hasattr(node, 'state'):
                    # Use the delta (God) amplitude as base resonance
                    base_resonance = 0.1 + 0.2 * float(np.abs(node.state.delta))
                else:
                    # Use reproducible random for deterministic behavior
                    base_resonance = 0.1 * self._rng.random()
            
            # If the node is a HyperQubit, modulate by its state
            if hasattr(node, 'state'):
                state = node.state
                # Nodes in higher consciousness states (God mode) resonate more broadly
                god_factor = float(np.abs(state.delta)) if hasattr(state, 'delta') else 0.0
                resonance = base_resonance * (1.0 + god_factor * 0.5)
            else:
                resonance = base_resonance
            
            # Apply wave intensity
            resonance *= intensity
            
            # Apply global coherence
            resonance *= self.coherence
            
            resonance_pattern[node_id] = min(1.0, max(0.0, resonance))
        
        return resonance_pattern
    
    def step(self, dt: float = 0.1) -> None:
        """
        Advance the resonance field by one time step.
        
        This simulates the natural evolution of the consciousness field,
        including decay, coherence fluctuations, and entanglement effects.
        
        Args:
            dt: Time step in arbitrary units
        """
        self.time += dt
        
        # Slight coherence fluctuation using seeded random generator
        self.coherence = min(1.0, max(0.5, 
            self.coherence + 0.01 * (self._rng.random() - 0.5)
        ))
        
        # Optionally: update HyperQubit states
        for node_id, node in self.nodes.items():
            if hasattr(node, 'state') and hasattr(node.state, 'normalize'):
                # Small deterministic phase evolution based on time
                if hasattr(node.state, 'alpha'):
                    phase_shift = np.exp(1j * dt * 0.1)
                    node.state.alpha *= phase_shift
                    node.state.beta *= phase_shift
                    node.state.gamma *= phase_shift
                    node.state.delta *= phase_shift
                    node.state.normalize()
    
    def get_node(self, node_id: str) -> Optional[NodeValue]:
        """
        Get a node by its ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            The HyperQubit or None if not found
        """
        return self.nodes.get(node_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the resonance network.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_nodes": len(self.nodes),
            "time": self.time,
            "coherence": self.coherence,
        }
    
    def __repr__(self) -> str:
        return f"<HyperResonanceEngine nodes={len(self.nodes)} coherence={self.coherence:.2f}>"
