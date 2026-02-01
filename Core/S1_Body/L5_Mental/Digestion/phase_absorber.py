"""
[Project Elysia] Phase Absorber
===============================
Absorbs digested CausalNodes into the 21D Qualia manifold.
Phase Backpropagation ensures integration with existing knowledge.
"""

import sys
import torch
from typing import List

root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.S1_Body.L5_Mental.Digestion.universal_digestor import CausalNode
from Core.S1_Body.L1_Foundation.Foundation.Graph.torch_graph import get_torch_graph
from Core.S1_Body.Tools.Scripts.plasticity_log import plasticity_logger


class PhaseAbsorber:
    """
    Absorbs CausalNodes into the 21D phase manifold.
    "흡수 (Absorption)" - Permanent brain structure modification.
    """
    
    def __init__(self):
        self.graph = get_torch_graph()
        self.learning_rate = 0.05  # Phase adjustment rate
    
    def absorb(self, nodes: List[CausalNode]) -> int:
        """
        Absorb a list of CausalNodes into the graph.
        Returns the number of nodes successfully absorbed.
        """
        absorbed_count = 0
        
        for node in nodes:
            try:
                # 1. Add node to graph if not exists
                if node.concept not in self.graph.id_to_idx:
                    self.graph.add_node(node.concept)
                
                node_idx = self.graph.id_to_idx[node.concept]
                
                # 2. Apply Phase Backpropagation to adjust Qualia
                if node.qualia_hint and len(node.qualia_hint) == 7:
                    # Use hint to guide phase adjustment
                    hint = torch.tensor(node.qualia_hint, device=self.graph.device)
                    with torch.no_grad():
                        current = self.graph.qualia_tensor[node_idx]
                        error = hint - current
                        self.graph.qualia_tensor[node_idx] += error * self.learning_rate
                else:
                    # Default: increase resonance (index 2)
                    with torch.no_grad():
                        self.graph.qualia_tensor[node_idx, 2] += self.learning_rate
                
                # 3. Note: Edges are handled at graph level via adjacency tensors
                # For now, we just log the intended relations
                if node.relations:
                    plasticity_logger.log_event(
                        "RELATION",
                        {"node": node.concept, "targets": node.relations[:3]},
                        0.01
                    )
                
                # 4. Log the structural change
                plasticity_logger.log_event(
                    "LTP",
                    {"node": node.concept, "source": node.source_chunk_id},
                    self.learning_rate
                )
                
                absorbed_count += 1
                
            except Exception as e:
                print(f"⚠️ Failed to absorb {node.concept}: {e}")
        
        return absorbed_count


# Singleton
_phase_absorber = None

def get_phase_absorber() -> PhaseAbsorber:
    global _phase_absorber
    if _phase_absorber is None:
        _phase_absorber = PhaseAbsorber()
    return _phase_absorber


if __name__ == "__main__":
    print("🧬 Testing Phase Absorber...")
    
    # Create test nodes
    test_nodes = [
        CausalNode(
            node_id="test_1",
            concept="Sovereignty",
            relations=["Freedom", "Will"],
            qualia_hint=[0.8, 0.2, 0.9, 0.5, 0.3, 0.7, 0.6]
        ),
        CausalNode(
            node_id="test_2",
            concept="Learning",
            relations=["Experience", "Growth"],
            qualia_hint=[0.6, 0.4, 0.8, 0.6, 0.5, 0.5, 0.7]
        )
    ]
    
    absorber = get_phase_absorber()
    count = absorber.absorb(test_nodes)
    
    print(f"✅ Absorbed {count} nodes into 21D manifold.")
    print("🎉 Phase Absorber operational!")
