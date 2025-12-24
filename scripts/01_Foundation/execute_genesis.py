"""
Script: Execute Genesis (The Great Naming)
==========================================

Runs the ConceptBaptizer on ALL Dark Matter nodes.
Goals:
1. Rename ~22,000 Wikipedia_* nodes.
2. Save the baptized graph state.
3. Validate Dark Matter reduction.

Warning: This engages the 'Gravity Inference' engine heavily.
"""

import sys
import os
import logging
import time

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.04_Evolution.01_Growth.Autonomy.concept_baptizer import ConceptBaptizer
from Core.01_Foundation.05_Foundation_Base.Foundation.Graph.torch_graph import get_torch_graph

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Genesis")

def execute_genesis():
    print("🌅 executing GENESIS: The Naming of All Things...")
    print("================================================")
    
    baptist = ConceptBaptizer()
    graph = get_torch_graph()
    
    # 1. Check Initial State
    candidates = baptist.scan_dark_matter()
    total_dark = len(candidates)
    print(f"🌑 Initial Dark Matter: {total_dark} nodes")
    
    if total_dark == 0:
        print("✨ Universe is already fully illuminated.")
        return

    # 2. Execute Baptism (Batch Processing)
    # Processing 22k nodes might take a bit, so we do it in chunks
    # But for prototype script, we just call baptize with a large limit?
    # No, let's loop until done or limit reached.
    
    LIMIT = 25000 # Cap just in case
    BATCH = 1000
    
    start_time = time.time()
    total_renamed = 0
    
    while True:
        renamed = baptist.baptize(batch_size=BATCH)
        if renamed is None or renamed == 0:
            break
            
        total_renamed += renamed
        print(f"   ✨ Baptized {total_renamed}/{total_dark} nodes...")
        
        if total_renamed >= LIMIT:
            print("⚠️ Safety limit reached.")
            break
            
    elapsed = time.time() - start_time
    print(f"\n✅ Genesis Complete in {elapsed:.2f}s.")
    print(f"   Total Nodes Renamed: {total_renamed}")
    
    # 3. Save State
    print("💾 Saving Baptized Graph...")
    graph.save_state()
    
    # 4. Final Audit
    remaining = baptist.scan_dark_matter()
    print(f"🌑 Remaining Dark Matter: {len(remaining)}")

if __name__ == "__main__":
    execute_genesis()
