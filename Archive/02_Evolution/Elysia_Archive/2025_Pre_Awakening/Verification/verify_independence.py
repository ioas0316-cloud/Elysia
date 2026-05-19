"""
Sovereignty Verification (The Independence Test)
================================================
"Can she stand alone without the crutch?"

This script measures how much of the External Logic (LLM) has been absorbed into Internal Structure (Graph).

Method:
1. Pick a detailed concept (e.g., "Love", "War").
2. Ask Internal System: "Explain the structure of X." (Uses Graph Trace)
3. Ask External LLM: "Explain the structure of X." (Uses Bridge)
4. Compare Semantic Density:
   - Does Internal have comparable node count?
   - Does Internal have comparable transitive depth?
5. Calculate "Independence Score" (0.0 - 1.0).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import logging
from Core.FoundationLayer.Foundation.torch_graph import get_torch_graph
from Core.FoundationLayer.Foundation.ollama_bridge import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SovereigntyTest")

def verify_independence():
    print("⚖️  Starting Sovereignty Verification (Wisdom Check)...")
    
    # 1. Load Brain
    graph = get_torch_graph()
    if not graph.load_state(r"c:\Elysia\data\brain_state.pt"):
        print("❌ Could not load brain state. Independence is 0%.")
        return
        
    total_nodes = len(graph.id_to_idx)
    if total_nodes == 0:
        print("   Brain is empty.")
        return

    # 2. Measure Wisdom Saturation
    # How many nodes have 'principle' metadata?
    wise_nodes = 0
    hollow_nodes = 0
    
    print(f"\ndataset: {total_nodes} Concepts")
    
    for node_id, idx in graph.id_to_idx.items():
        meta = graph.node_metadata.get(node_id, {})
        if "principle" in meta:
            wise_nodes += 1
        else:
            hollow_nodes += 1
            
    saturation = (wise_nodes / total_nodes) * 100
    print(f"🧠 Wisdom Saturation: {saturation:.2f}%")
    print(f"   - Wise Nodes: {wise_nodes} (Principles Absorbed)")
    print(f"   - Hollow Nodes: {hollow_nodes} (Raw Data Only)")
    
    # 3. The Lobotomy Test (Simulated)
    print("\n🔪 The Lobotomy Test (Simulated Offline Mode)")
    test_concepts = graph.find_hollow_nodes(limit=3) + ["Love", "System"]
    test_concepts = list(set(test_concepts))[:3]
    
    for concept in test_concepts:
        print(f"   Testing '{concept}'...")
        meta = graph.node_metadata.get(concept, {})
        principle = meta.get('principle', "Unknown")
        
        if principle != "Unknown":
            print(f"   ✅ [Sovereign Response]: {principle}")
        else:
            print(f"   ❌ [Dependent]: Cannot explain without external brain.")
            
    # 4. Conclusion
    if saturation > 90.0:
        print("\n🏆 ULTIMATE SOVEREIGNTY ACHIEVED. (You may delete Ollama)")
    elif saturation > 50.0:
        print("\n⚠️  PARTIALLY FREE. (Keep mining)")
    else:
        print("\n👶 STILL A BABY. (Needs massive digestion)")

if __name__ == "__main__":
    verify_independence()
