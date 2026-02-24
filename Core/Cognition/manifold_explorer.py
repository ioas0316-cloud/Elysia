"""
Manifold Explorer: Diving into the Inner Cosmos
===============================================

"The surface is a mirror; the depth is the truth."

This script explores the Knowledge Graph, showing the recursive 
'inner_cosmos' and linguistic resonance of concepts.
"""

import sys
import json
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from Core.Cognition.kg_manager import KGManager

def explore(node_id: str):
    kg = KGManager()
    node = kg.get_node(node_id)
    
    if not node:
        print(f"❌ '{node_id}' not found in the manifold.")
        return

    print(f"\n🌌 [EXPLORE] Node: {node_id.upper()}")
    print(f"   Layer: {node.get('layer', 'surface')} (Confidence: {node.get('layer_confidence', 0.0)})")
    
    # 1. External Relations
    print("\n   🔗 [EXTERNAL RELATIONS]")
    edges = kg.kg.get('edges', [])
    related = [e for e in edges if e['source'] == node_id or e['target'] == node_id]
    for r in related:
        dir = "->" if r['source'] == node_id else "<-"
        target = r['target'] if r['source'] == node_id else r['source']
        print(f"      {node_id} {dir} [{r['relation']}] -> {target}")

    # 2. Inner Cosmos (The "Why")
    cosmos = node.get("inner_cosmos", {})
    if cosmos and cosmos.get("nodes"):
        print("\n   🪐 [INNER COSMOS] (Internal Universe)")
        print(f"      Depth: {cosmos.get('depth', 0)}")
        print(f"      Constituent Nodes: {list(cosmos['nodes'].keys())[:10]}...")
        print(f"      Constitutive Edges: {len(cosmos.get('edges', []))}")
        
        # Show a few constitutive relations
        for edge in cosmos.get('edges', [])[:5]:
            print(f"         {edge['source']} -> [{edge['relation']}] -> {edge['target']}")
    else:
        print("\n   ⚪ [INNER COSMOS] Still a singular point (Un-digested).")

    # 3. Resonance
    resonates = [e['target'] for e in edges if e['source'] == node_id and e['relation'] == "resonates_with"]
    if resonates:
        print("\n   💎 [RESONANCE BRIDGES]")
        for target in resonates:
            print(f"      {node_id} <≈≈≈> {target} (Unified Concept)")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "water"
    explore(target)
