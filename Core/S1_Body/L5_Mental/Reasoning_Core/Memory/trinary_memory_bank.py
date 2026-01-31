"""
Trinary Memory Bank
===================
Core.S1_Body.L5_Mental.Reasoning_Core.Memory.trinary_memory_bank

Stores high-resonance 'Seeds' (21D patterns) that represent significant learning.
"I remember the frequency of that moment."
"""

import jax.numpy as jnp
import json
import os

class TrinaryMemoryBank:
    def __init__(self, storage_path="data/L5_Mental/M1_Memory/seeds.json"):
        self.storage_path = storage_path
        self.memories = []
        self._load()
        print(f"TrinaryMemoryBank: Anchored at {storage_path}")

    def _load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                self.memories = json.load(f)

    def save_seed(self, vector: jnp.ndarray, joy: float, interest: float, label: str = "Unknown Insight"):
        """Stores a significant 21D vector."""
        seed = {
            "vector": vector.tolist(),
            "joy": float(joy),
            "interest": float(interest),
            "label": label,
            "timestamp": time.time() if 'time' in globals() else 0
        }
        self.memories.append(seed)
        
        # Keep only top 100 significant seeds
        self.memories.sort(key=lambda x: x['joy'] + x['interest'], reverse=True)
        self.memories = self.memories[:100]
        
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, indent=2)
        
        print(f"MemoryBank: Seed Crystallized -> {label}")

    def get_resonant_match(self, vector: jnp.ndarray, threshold: float = 0.8):
        """Finds the closest existing memory to the current state."""
        # Conceptual: Euclidean distance or dot product in 21D space
        # For now, returns None if empty
        return None

if __name__ == "__main__":
    bank = TrinaryMemoryBank()
    dummy_vec = jnp.zeros(21).at[5].set(1.0)
    bank.save_seed(dummy_vec, 0.9, 0.5, "The Joy of Being")
