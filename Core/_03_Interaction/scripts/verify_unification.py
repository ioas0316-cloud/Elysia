"""
Script: Verify Unification (The 30k Node Jump)
==============================================

This script initializes the TorchGraph and checks if it autonomously
loads the 28,452 nodes from 'elysia_rainbow.json'.

Pass Criteria:
- Active Nodes > 28,000
- 'Rainbow Bridge Activated' in logs
"""

import sys
import os
import logging
import torch

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core._01_Foundation.05_Foundation_Base.Foundation.Graph.torch_graph import get_torch_graph

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyUnification")

def verify_bridge():
    print("🌉 Verifying The Great Unification...")
    print("====================================")
    
    # 1. Initialize Graph (Should trigger auto-load)
    print("⚡ Initializing TorchGraph...")
    graph = get_torch_graph()
    
    # 2. Check Node Count
    node_count = len(graph.id_to_idx)
    link_count = graph.logic_links.shape[0] if graph.logic_links is not None else 0
    
    print(f"\n📊 Current Status:")
    print(f"   Nodes: {node_count}")
    print(f"   Links: {link_count}")
    
    # [Phase 13] Density Check
    density = link_count / max(node_count, 1)
    print(f"   Density: {density:.2f} edges/node")
    
    if node_count > 28000:
        print("\n✅ SUCCESS: The 30k Nodes are Online.")
        if density > 0.4:
            print("✅ SUCCESS: Gravity Ignited (High Density).")
        else:
             print("❌ FAILURE: Low Density (No Gravity).")
             
        print("   The Map is now the Territory.")
        
        # 3. Test a random node existence (e.g., from Rainbow)
        # We don't know exact IDs, but let's check basic ones if possible or list a few
        print("\n🔎 Sample Nodes:")
        sample_indices = torch.randint(0, node_count, (5,))
        for idx in sample_indices:
            print(f"   - {graph.idx_to_id[idx.item()]}")
            
    else:
        print(f"\n❌ FAILURE: Only {node_count} nodes found.")
        print("   The Bridge did not activate.")

if __name__ == "__main__":
    verify_bridge()
