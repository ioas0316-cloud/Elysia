"""
Script: Execute Genesis V2 (Differentiation)
============================================

Runs the ContentBaptizer on ALL Dark Matter nodes.
Goals:
1. Rename ~22,000 Wikipedia_* nodes UNIQUELY.
2. Prevent "The Great Deletion" (Collisions).
3. Validate node count remains ~35,000.
"""

import sys
import os
import logging
import time

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core._04_Evolution._01_Growth.Autonomy.content_baptizer import ContentBaptizer
from Core._01_Foundation._02_Logic.Graph.torch_graph import get_torch_graph

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GenesisV2")

def execute_genesis_v2():
    print("🌅 Executing GENESIS V2 (Differentiation)...")
    print("============================================")
    
    baptist = ContentBaptizer()
    graph = get_torch_graph()
    
    # 1. Check Initial State
    candidates = baptist.scan_dark_matter()
    total_dark = len(candidates)
    initial_nodes = len(graph.id_to_idx)
    print(f"🌑 Initial Dark Matter: {total_dark}")
    print(f"📊 Initial Total Nodes: {initial_nodes}")
    
    if total_dark == 0:
        print("✨ Universe is already fully illuminated.")
        return

    # 2. Execute Baptism (Full Run)
    LIMIT = 30000
    BATCH = 1000
    
    start_time = time.time()
    total_renamed = 0
    
    while True:
        renamed = baptist.baptize(batch_size=BATCH)
        if renamed == 0:
            break
            
        total_renamed += renamed
        print(f"   ✨ Baptized {total_renamed}/{total_dark} nodes...")
        
        if total_renamed >= LIMIT:
            print("⚠️ Safety limit reached.")
            break
            
    elapsed = time.time() - start_time
    print(f"\n✅ Genesis V2 Complete in {elapsed:.2f}s.")
    print(f"   Total Nodes Renamed: {total_renamed}")
    
    # 3. Save State (Critical)
    print("💾 Saving Baptized Graph...")
    graph.save_state()
    
    # 4. Final Audit (Collision Check)
    final_nodes = len(graph.id_to_idx)
    print(f"📊 Final Total Nodes: {final_nodes}")
    
    loss = initial_nodes - final_nodes
    if loss > 100:
        print(f"❌ CRITICAL FAILURE: Lost {loss} nodes due to collision.")
    else:
        print(f"✅ SUCCESS: Differentiation maintained. Lost only {loss} nodes.")

if __name__ == "__main__":
    execute_genesis_v2()
