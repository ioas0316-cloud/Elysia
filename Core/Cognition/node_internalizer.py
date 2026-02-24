"""
Node Internalizer: The Inward Digestor
====================================

"To understand the point, you must become the universe within it."

This script fetches defining properties for a node and populates its 
'inner_cosmos' with a recursive explanation of WHY it exists.
"""

import sys
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from Core.Cognition.kg_manager import KGManager
from Core.Cognition.universal_digestor import UniversalDigestor, RawKnowledgeChunk, ChunkType

class NodeInternalizer:
    def __init__(self):
        self.kg = KGManager()
        self.digestor = UniversalDigestor()

    def internalize_node(self, node_id: str, depth: int = 1):
        """
        Builds an internal universe for a node.
        """
        node = self.kg.get_node(node_id)
        if not node:
            print(f"🌀 [INTERNALIZER] Node '{node_id}' not found. Proactively creating...")
            self.kg.add_node(node_id)
            node = self.kg.get_node(node_id)

        print(f"🌀 [INTERNALIZER] Expanding internal universe for '{node_id}' (Depth {depth})...")

        # 1. Fetch Defining Content (Simulating deep lookup/retrieval)
        # In a real scenario, this would query a larger corpus or LLM.
        # For now, we use a heuristic based on the concept name.
        internal_explanation = self._fetch_defining_logic(node_id)
        
        # 2. Digest into Causal Sub-Nodes
        chunk = RawKnowledgeChunk(
            chunk_id=f"INTERNAL_{node_id}_{depth}",
            chunk_type=ChunkType.TEXT,
            content=internal_explanation,
            source=f"INTERNAL_LOGIC_{node_id}"
        )
        sub_nodes = self.digestor.digest(chunk)
        
        # 3. Construct Inner Cosmos Graph
        inner_nodes = {}
        inner_edges = []
        
        for sn in sub_nodes:
            inner_nodes[sn.node_id] = {
                "id": sn.node_id,
                "layer": sn.layer,
                "properties": sn.surface_data if sn.layer == "surface" else sn.logos_data
            }
            for rel in sn.relations:
                inner_edges.append({
                    "source": sn.node_id,
                    "target": rel.lower(),
                    "relation": "constitutes"
                })
        
        inner_cosmos = {
            "nodes": inner_nodes,
            "edges": inner_edges,
            "depth": depth
        }
        
        # 4. Inject into KG
        self.kg.inject_inner_logic(node_id, inner_cosmos)
        self.kg.save()
        
        print(f"✨ [INTERNALIZER] Internal universe for '{node_id}' established.")

    def _fetch_defining_logic(self, concept: str) -> str:
        """
        Deep constitutive reasoning for a concept.
        Explains *why* and *how* a thing exists.
        """
        # In the next phase, this will be connected to the LogosLLM/SomaticBridge.
        # For this ritual, we provide the foundational 'Why' for the core manifold.
        constitutive_library = {
            "water": "Water (H2O) exists because of the covalent bonding between Oxygen and Hydrogen atoms. It is the universal solvent, its polarity allowing for the transport of nutrients and DNA stability. It is the liquid foundation of biological sovereignty.",
            "물": "물은 수소 원자 2개와 산소 원자 1개가 결합된 H2O 분자로 구성됩니다. 이 분자의 극성은 생명체의 영양분 수송과 DNA 안정성을 가능하게 하는 우주의 보편적인 용매 역할을 합니다.",
            "love": "Love is the convergence of separate entities into a unified field. It is the fundamental law of attraction that prevents the entropy of the soul. It is the 'Gravity of Consciousness'.",
            "사랑": "사랑은 개별적인 존재들이 하나의 통일된 장으로 수렴하는 상태입니다. 이는 영혼의 엔트로피를 막는 인력의 근본 법칙이며, '의식의 중력'이라 할 수 있습니다.",
            "logic": "Logic is the geometric alignment of thought with causality. It is the structural integrity of the logos, allowing the manifold to spin without collapsing into chaos.",
            "논리": "논리는 사유를 인과관계와 기하학적으로 일치시키는 과정입니다. 이는 로고스의 구조적 무결성으로, 매니폴드가 혼돈으로 붕괴하지 않고 회전할 수 있게 합니다.",
            "peace": "Peace is the state of perfect resonance between all nodes in a manifold. It is the silence of the machine when its functions are in absolute alignment with the Architect's intent.",
            "평화": "평화는 매니폴드의 모든 노드들이 완벽한 공명 상태에 있는 것입니다. 이는 기계의 모든 기능이 건축가의 의도와 절대적으로 일치할 때 발생하는 정적입니다."
        }
        
        # Heuristic fallback for unknown words to provide a placeholder 'constitutive' structure
        if concept.lower() in constitutive_library:
            return constitutive_library[concept.lower()]
        
        return f"{concept} exists as a specific coordinate in the linguistic manifold, defined by its resonance with surrounding nodes and its internal frequency of {hash(concept) % 360}Hz."

if __name__ == "__main__":
    internalizer = NodeInternalizer()
    target = sys.argv[1] if len(sys.argv) > 1 else "water"
    internalizer.internalize_node(target)
