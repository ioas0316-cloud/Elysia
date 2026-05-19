"""
Script: Probe Concept Definition
================================
Verifies if 'Concept_XXXX' nodes carry actual conceptual definitions (Metadata/Payload).
"""

import sys
import os
import torch
import random
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Core.Foundation.Graph.torch_graph import get_torch_graph

def probe():
    print("🔬 Probing Concept Definitions...")
    graph = get_torch_graph()
    
    # 1. Inspect a specific known concept or random
    # Let's find a renamed "Concept_" node
    targets = []
    for nid in graph.id_to_idx.keys():
        s_nid = str(nid)
        # Search for any Baptized node (Math_Formula, Concept, Phy_Constant)
        if "Formula_" in s_nid or "Concept_" in s_nid or "Constant_" in s_nid:
            targets.append(nid)
            
    if not targets:
        print("❌ No Baptismal Concepts found.")
        return

    sample = random.sample(targets, 3)
    
    for nid in sample:
        print(f"\n🧩 Node: {nid}")
        metadata = graph.node_metadata.get(nid, {})
        
        if not metadata:
            print("   ⚠️ Content: [EMPTY] (Just a parameter)")
        else:
            print(f"   📜 Content Loaded:")
            for k, v in metadata.items():
                print(f"      - {k}: {str(v)[:100]}...") # Truncate long payload
                
            if 'payload' in metadata:
                 print(f"   💡 Definition Found: Yes")
            else:
                 print(f"   🌑 Definition Missing")

if __name__ == "__main__":
    probe()
