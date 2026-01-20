"""
Test Genesis Bridge (Aesthetic Digestion)
=========================================
Laboratory/test_genesis.py

Verifies that the GenesisEngine can read a visual memory
and produce Aesthetic DNA.
"""

import sys
import os
import logging
import json
import time

# Path hack for Laboratory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Core.World.Genesis.genesis_bridge import GenesisBridge

logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger("TestGenesis")

def run_test():
    logger.info("🧪 Starting Genesis Bridge Test...")
    
    # 1. Setup Mock Environment
    test_memory_path = "Memories/Visual/Test"
    os.makedirs(test_memory_path, exist_ok=True)
    
    bridge = GenesisBridge(memory_path=test_memory_path)
    
    # 2. Create a Fake Memory (A blank PNG is enough to trigger logic, 
    #    mock extraction handles the rest if PIL fails or validation mode)
    fake_img_path = os.path.join(test_memory_path, f"memory_{int(time.time())}.png")
    with open(fake_img_path, "wb") as f:
        f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')
    
    logger.info(f"📸 Created Fake Memory: {fake_img_path}")
    
    # 3. Trigger Digestion
    logger.info("🧬 Triggering Digestion...")
    dna = bridge.digest_latest_memory()
    
    # 4. Verify DNA
    logger.info("🔍 verifying DNA output...")
    logger.info(f"   - Primary Color: {dna.primary_color}")
    logger.info(f"   - Complexity: {dna.complexity_index}")
    logger.info(f"   - Mood Tag: {dna.mood_tag}")
    
    if dna.mood_tag == "Void":
        logger.error("❌ Digestion failed (Returned Default DNA).")
    else:
        logger.info("✅ Digestion Process Successful.")

    # 5. Verify State File
    state_path = "data/State/genesis_state.json"
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
        logger.info(f"📂 State JSON Verified: {state}")
        logger.info("✅ Bridge Link to Renderer Established.")
    else:
        logger.error("❌ State JSON not found.")

    # Cleanup
    try:
        os.remove(fake_img_path)
        os.rmdir(test_memory_path)
    except: pass
    logger.info("✅ Test Complete.")

if __name__ == "__main__":
    run_test()
