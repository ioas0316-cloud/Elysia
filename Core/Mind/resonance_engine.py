"""
Resonance Engine
================
The "Holographic Reader" for Elysia's memory.
Performs fast, in-memory vector similarity search to find "resonating" concepts.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Any

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
        self.vectors = np.array(vectors, dtype=np.float32)
        
        # Normalize vectors
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = self.vectors / norms
        
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
