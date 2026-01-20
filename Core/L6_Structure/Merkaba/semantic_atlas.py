"""
Semantic Atlas (The Naming Engine)
==================================
Core.L6_Structure.Merkaba.semantic_atlas

"To name a thing is to know its soul."

This module manages the mapping between neural hubs and semantic concepts.
It allows Elysia to categorize her weights into functional domains.
"""

import json
import os
import logging
from typing import Dict, Any, List, Set

logger = logging.getLogger("Elysia.Merkaba.Atlas")

class SemanticAtlas:
    def __init__(self, atlas_path: str = "data/Logs/topology_maps/semantic_atlas.json"):
        self.atlas_path = atlas_path
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.atlas_path):
            with open(self.atlas_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "version": "1.0",
            "concepts": {
                "LOGIC": [], # Tensors related to pure reasoning
                "MATH": [],  # Tensors related to arithmetic
                "CODE": [],  # Tensors related to programming
                "IDENTITY": [], # Tensors related to self-awareness/reflexive prompts
                "AESTHETIC": [], # Tensors related to visual/style
                "CORE": []   # Essential 'Root' tensors
            },
            "mapped_tensors": {}
        }

    def save(self):
        os.makedirs(os.path.dirname(self.atlas_path), exist_ok=True)
        with open(self.atlas_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def tag_tensor(self, tensor_name: str, concepts: List[str], strength: float = 1.0):
        """
        Associates a tensor with one or more concepts.
        """
        if tensor_name not in self.data["mapped_tensors"]:
            self.data["mapped_tensors"][tensor_name] = []
        
        for concept in concepts:
            concept = concept.upper()
            if concept not in self.data["concepts"]:
                self.data["concepts"][concept] = []
            
            # Update mapped_tensors record
            entry = {"concept": concept, "strength": strength}
            if entry not in self.data["mapped_tensors"][tensor_name]:
                self.data["mapped_tensors"][tensor_name].append(entry)
            
            # Update concepts list
            if tensor_name not in self.data["concepts"][concept]:
                self.data["concepts"][concept].append(tensor_name)
        
        logger.info(f"🏷️ Tagged {tensor_name} with {concepts}")

    def get_tensors_by_concept(self, concept: str) -> List[str]:
        return self.data["concepts"].get(concept.upper(), [])

    def get_concepts_of_tensor(self, tensor_name: str) -> List[Dict[str, Any]]:
        return self.data["mapped_tensors"].get(tensor_name, [])
    
    def get_principle_vector(self, tensor_name: str, dim: int = 2048) -> "np.ndarray":
        """
        Generates a synthetic 'principle vector' based on the tensor's concept mappings.
        This allows operation WITHOUT original model weights.
        
        The vector encodes the semantic meaning of the tensor based on its concepts,
        not its actual learned weights.
        """
        import numpy as np
        
        concepts = self.get_concepts_of_tensor(tensor_name)
        
        if not concepts:
            # Unknown tensor - return neutral vector
            return np.zeros(dim)
        
        # Create a deterministic hash-based vector from the tensor name
        # This ensures consistency across runs
        seed = sum(ord(c) for c in tensor_name) % (2**31)
        rng = np.random.RandomState(seed)
        base = rng.randn(dim).astype(np.float32)
        
        # Modulate by concept strengths
        total_strength = sum(c.get("strength", 1.0) for c in concepts)
        base *= (total_strength / max(1, len(concepts)))
        
        # Normalize
        norm = np.linalg.norm(base)
        if norm > 0:
            base /= norm
        
        return base


if __name__ == "__main__":
    atlas = SemanticAtlas()
    print("Semantic Atlas: Engine Ready.")
