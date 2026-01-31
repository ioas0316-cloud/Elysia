"""
Test Topology Evolution (Phase 19.2 Verification)
=================================================
Laboratory/test_topology_evolution.py

Verifies that the TopologyManager correctly reorders memory blocks
based on access frequency (Heatmap).
"""

import sys
import os
import logging
import numpy as np

# Path hack for Laboratory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Core.1_Body.L2_Metabolism.Memory.memory_collapse import TopologyManager

logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger("TestTopology")

def run_test():
    logger.info("🧪 Starting Topology Evolution Test...")
    
    # 1. Initialize Mock Memory (100 slots)
    raw_memory = np.zeros(100, dtype=np.float32)
    manager = TopologyManager(raw_memory)
    
    # 2. Allocation
    # Ideally: Vision(1.0), Audio(2.0), Logic(3.0)
    manager.allocate("Vision_Cortex", 10) 
    manager.allocate("Audio_Cortex", 10) 
    manager.allocate("Logic_Core", 10)
    
    # Inscribe Data
    raw_memory[0:10] = 1.0   # Vision
    raw_memory[10:20] = 2.0  # Audio
    raw_memory[20:30] = 3.0  # Logic
    
    logger.info("🧱 [Initial State]")
    logger.info(f"   - Vision (Offset 0): {raw_memory[0]}")
    logger.info(f"   - Audio (Offset 10): {raw_memory[10]}")
    logger.info(f"   - Logic (Offset 20): {raw_memory[20]}")
    
    # 3. Simulate Neuroplasticity (Usage Pattern)
    logger.info("🔥 [Simulation] Heavy firing in Logic Core...")
    for _ in range(100): manager.access("Logic_Core")
    for _ in range(10):  manager.access("Vision_Cortex")
    # Audio is untouched (Cold)
    
    # 4. Trigger Evolution
    manager.evolve_topology()
    
    # 5. Verification
    # Logic (3.0) should be at Offset 0 (Hottest)
    # Vision (1.0) should be at Offset 10 (Warm)
    # Audio (2.0) should be at Offset 20 (Cold)
    
    new_first_block_val = raw_memory[0]
    
    logger.info("🧠 [Evolved State]")
    logger.info(f"   - Block at Offset 0: {new_first_block_val}")
    logger.info(f"   - Block at Offset 10: {raw_memory[10]}")
    
    if new_first_block_val == 3.0:
        logger.info("🚀 [Topology] SUCCESS! Logic Core moved to Prime Offset.")
    else:
        logger.error(f"❌ Topology Validation Failed. Expected 3.0 at Offset 0, got {new_first_block_val}")

if __name__ == "__main__":
    run_test()
