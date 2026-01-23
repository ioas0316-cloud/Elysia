"""
TOPOLOGICAL HOLOGRAM: The Liquid Store
========================================
Core.L6_Structure.M3_Sphere.topological_hologram

"Memories are not boxes; they are phase patterns in the abyss."
"""

import torch
from typing import List, Dict, Any, Optional

class TopologicalHologram:
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.memory_tensor = torch.zeros((0, dimension))
        self.index_map = [] # List of unique IDs/Keys
        
    def imprint(self, key: str, vector: torch.Tensor):
        """
        [WAVING]
        Imprints a new memory into the holographic field.
        If key exists, it merges (interference) rather than overwrites.
        """
        if vector.shape[-1] != self.dimension:
            raise ValueError(f"Vector dimension must be {self.dimension}")
            
        vector = vector.view(1, -1)
        
        if key in self.index_map:
            idx = self.index_map.index(key)
            # Interferential Merging (Constructive Resonance)
            current = self.memory_tensor[idx]
            self.memory_tensor[idx] = (current + vector) / 2.0
        else:
            self.index_map.append(key)
            self.memory_tensor = torch.cat([self.memory_tensor, vector], dim=0)
            
    def retrieve(self, query_vector: torch.Tensor, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        [GRAVITATIONAL RETRIEVAL]
        Uses cosine similarity (gravity) to pull resonant memories.
        """
        if self.memory_tensor.shape[0] == 0:
            return []
            
        query_vector = query_vector.view(1, -1)
        similarities = torch.nn.functional.cosine_similarity(self.memory_tensor, query_vector)
        
        values, indices = torch.topk(similarities, min(top_k, len(self.index_map)))
        
        results = []
        for val, idx in zip(values, indices):
            results.append({
                "key": self.index_map[int(idx)],
                "resonance": float(val),
                "vector": self.memory_tensor[int(idx)]
            })
        return results

    def get_status(self) -> str:
        return f"TopologicalHologram: {len(self.index_map)} Memories Sealed | Field Shape: {self.memory_tensor.shape}"
