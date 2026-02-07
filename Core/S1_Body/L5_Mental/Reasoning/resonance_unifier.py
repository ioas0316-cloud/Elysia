"""
Resonance Unifier: Bridging the Linguistic Gap
=============================================

"One essence, many names."

This script identifies nodes that refer to the same universal concept 
(e.g., 'water' and '물') and ensures they share or mirror 
the same internal universe (inner_cosmos).
"""

import sys
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from Core.S1_Body.L5_Mental.Memory.kg_manager import KGManager

class ResonanceUnifier:
    def __init__(self):
        self.kg = KGManager()

    def unify_resonance(self):
        """
        Identify and bridge cross-language pairs.
        """
        # Mapping of universal concepts to their linguistic labels
        # In a more advanced version, this would be derived from embeddings or Satori.
        resonance_pairs = [
            ("water", "물"),
            ("love", "사랑"),
            ("peace", "평화"),
            ("truth", "진리"),
            ("freedom", "자유"),
            ("happiness", "행복"),
            ("logic", "논리"),
            ("fire", "불"),
            ("earth", "흙"),
            ("wind", "바람")
        ]

        print("🔗 [RESONANCE] Starting Unification...")

        for eng, kor in resonance_pairs:
            e_node = self.kg.get_node(eng)
            k_node = self.kg.get_node(kor)

            if e_node and k_node:
                print(f"🌉 [RESONANCE] Joining '{eng}' <-> '{kor}'")
                
                # 1. Share or Mirror Inner Cosmos
                # If one has an inner_cosmos and the other doesn't, copy it.
                e_cosmos = e_node.get("inner_cosmos", {})
                k_cosmos = k_node.get("inner_cosmos", {})

                if e_cosmos.get("nodes") and not k_cosmos.get("nodes"):
                    self.kg.inject_inner_logic(kor, e_cosmos)
                elif k_cosmos.get("nodes") and not e_cosmos.get("nodes"):
                    self.kg.inject_inner_logic(eng, k_cosmos)
                
                # 2. Add an explicit 'resonance' edge in the global KG
                self.kg.add_edge(eng, kor, "resonates_with", {"type": "linguistic_bridge"})
                self.kg.add_edge(kor, eng, "resonates_with", {"type": "linguistic_bridge"})

        self.kg.save()
        print("✨ [RESONANCE] Unification complete. Manifold is now unified across English and Korean.")

if __name__ == "__main__":
    unifier = ResonanceUnifier()
    unifier.unify_resonance()
