"""
[Project Elysia] Entropy Purger (LTD System)
=============================================
Phase 500: Excretion Layer - "배설 (Excretion)"
Removes redundant, contradictory, or low-value nodes via Long-Term Depression.
"""

import sys
from typing import List, Dict, Set
from collections import Counter

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.System.torch_graph import get_torch_graph
from Core.System.plasticity_log import plasticity_logger


class EntropyPurger:
    """
    배설 (Excretion) - Removes noise and redundancy from the cognitive manifold.
    Implements Long-Term Depression (LTD) for weakening unused paths.
    
    "불필요한 것은 버려야 필요한 것이 빛난다."
    """
    
    def __init__(self, 
                 redundancy_threshold: float = 0.95,
                 min_resonance: float = 0.1,
                 decay_rate: float = 0.01):
        self.redundancy_threshold = redundancy_threshold
        self.min_resonance = min_resonance
        self.decay_rate = decay_rate
        self.graph = get_torch_graph()
    
    def purge_redundant(self) -> int:
        """
        Remove nodes that are too similar to existing nodes (redundancy).
        Returns the number of purged nodes.
        """
        purged = 0
        node_ids = list(self.graph.id_to_idx.keys())
        
        # Find duplicate-like concepts (same lowercase)
        concept_counts = Counter([n.lower() for n in node_ids])
        duplicates = [c for c, count in concept_counts.items() if count > 1]
        
        for dup in duplicates:
            # Keep the first, mark others for decay
            matching = [n for n in node_ids if n.lower() == dup]
            for redundant in matching[1:]:  # Skip first (original)
                if redundant in self.graph.id_to_idx:
                    idx = self.graph.id_to_idx[redundant]
                    # Apply LTD: decay resonance
                    self.graph.qualia_tensor[idx, 2] -= self.decay_rate * 5
                    purged += 1
                    
                    plasticity_logger.log_event(
                        "LTD",
                        {"node": redundant, "reason": "redundancy"},
                        -self.decay_rate * 5
                    )
        
        return purged
    
    def purge_low_resonance(self) -> int:
        """
        Decay nodes with very low resonance (unused knowledge).
        Returns the number of affected nodes.
        """
        affected = 0
        
        if self.graph.qualia_tensor is None:
            return 0
        
        # Find nodes with resonance below threshold
        resonance_col = self.graph.qualia_tensor[:, 2]
        low_res_mask = resonance_col < self.min_resonance
        
        # Apply decay to low-resonance nodes
        import torch
        with torch.no_grad():
            self.graph.qualia_tensor[low_res_mask, 2] -= self.decay_rate
            affected = int(low_res_mask.sum().item())
        
        if affected > 0:
            plasticity_logger.log_event(
                "LTD",
                {"count": affected, "reason": "low_resonance"},
                -self.decay_rate
            )
        
        return affected
    
    def full_purge_cycle(self) -> Dict[str, int]:
        """
        Run a complete purge cycle.
        Returns statistics about purged/affected nodes.
        """
        print("\n🧹 [ENTROPY PURGER] Running full purge cycle...")
        
        redundant = self.purge_redundant()
        low_res = self.purge_low_resonance()
        
        stats = {
            "redundant_purged": redundant,
            "low_resonance_decayed": low_res,
            "total_affected": redundant + low_res
        }
        
        print(f"   - Redundant nodes decayed: {redundant}")
        print(f"   - Low-resonance nodes decayed: {low_res}")
        print(f"   ✅ Total affected: {stats['total_affected']}")
        
        return stats


# Singleton
_entropy_purger = None

def get_entropy_purger() -> EntropyPurger:
    global _entropy_purger
    if _entropy_purger is None:
        _entropy_purger = EntropyPurger()
    return _entropy_purger


if __name__ == "__main__":
    print("🧹 Testing Entropy Purger...")
    
    purger = get_entropy_purger()
    stats = purger.full_purge_cycle()
    
    print(f"\n🎉 Entropy Purger operational!")
    print(f"   Manifold entropy reduced by {stats['total_affected']} nodes.")
