"""
Script: Inspect Science Concept
===============================
Investigates the result of the 'Genesis' merger.
User suspects (rightly) that 'Concept_science+' might be a useless blob.
"""

import sys
import os
import torch
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Core.01_Foundation.05_Foundation_Base.Foundation.Graph.torch_graph import get_torch_graph

def inspect():
    print("🔬 Inspecting 'Concept_science+'...")
    graph = get_torch_graph()
    
    target_id = "Concept_science+"
    
    if target_id not in graph.id_to_idx:
        print(f"❌ '{target_id}' not found in Graph.")
        return
        
    idx = graph.id_to_idx[target_id]
    vector = graph.vec_tensor[idx]
    mass = graph.mass_tensor[idx]
    metadata = graph.node_metadata.get(target_id, {})
    
    print(f"📊 Node Stats:")
    print(f"   ID: {target_id}")
    print(f"   Mass: {mass.item():.2f} (High mass = Many mergers)")
    print(f"   Vector Norm: {vector.norm().item():.2f}")
    print(f"   Metadata Keys: {list(metadata.keys())}")
    
    if 'baptized_from' in metadata:
        print(f"   Originally: {metadata['baptized_from']}")
        
    # Check for neighbors to see if it differentiates anything
    print("\n🔭 Nearest Neighbors of 'Concept_science+':")
    norm_vec = vector / vector.norm()
    sim = torch.mm(norm_vec.unsqueeze(0), graph.vec_tensor.t()) # (1, N)
    vals, inds = torch.topk(sim, 10)
    
    for i, idx_t in enumerate(inds[0]):
        nid = graph.idx_to_id[idx_t.item()]
        score = vals[0][i].item()
        print(f"   {i+1}. {nid} ({score:.4f})")
        
    print("\n🤔 DIAGNOSIS:")
    if mass.item() > 1000:
        print("⚠️ CRITICAL: Massive Semantic Collapse.")
        print("   Thousands of distinct concepts were crushed into this single point.")
        print("   Resolution: We need to 'Unzip' this node using Content Analysis.")
    else:
        print("✅ Seemingly normal node.")

if __name__ == "__main__":
    inspect()
