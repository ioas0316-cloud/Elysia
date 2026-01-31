"""
VERIFICATION: Phase 14 - Cognitive Archeology Initial Excavation
==============================================================
"""

import logging
import os
from Core.L5_Mental.Reasoning_Core.Metabolism.topology_predator import CognitiveArcheologist
from Core.L5_Mental.Reasoning_Core.LLM.target_registry import TargetLLM, ModelType
from Core.L6_Structure.M1_Merkaba.Space.hypersphere_memory import HypersphereMemory

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Archeology.Verify")

def run_excavation():
    # 1. Prepare Environment
    memory = HypersphereMemory()
    archeologist = CognitiveArcheologist(memory_ref=memory)
    
    # 2. Define Target (Qwen2.5-0.5B)
    target = TargetLLM(
        id="qwen2.5:0.5b",
        name="Qwen2.5-0.5B Fossil",
        params="0.5B",
        type=ModelType.TEXT,
        tier=1,
        vram_myth="1.5GB",
        our_reality="398MB (mmap Archeology)"
    )
    
    # Path to the blob (identified earlier)
    fossil_path = r"C:\Users\USER\.ollama\models\blobs\sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515"
    
    if not os.path.exists(fossil_path):
        logger.error(f"❌ Fossil not found at {fossil_path}")
        return

    # 3. Excavate
    logger.info("🏺 Starting excavation of the Qwen2.5 era artifact...")
    discovery_map = archeologist.excavate(target, fossil_path)
    
    if discovery_map:
        logger.info("\n📜 EXCAVATION REPORT:")
        logger.info(f"  - Glimmers of Intent found: {len(discovery_map['intents'])}")
        logger.info(f"  - Clouds of Abstraction found: {len(discovery_map['abstractions'])}")
        
        # Look at one glimmer
        if discovery_map['intents']:
            sample = discovery_map['intents'][0]
            logger.info(f"\n✨ Sample Intent recovered from {sample['layer']}:")
            logger.info(f"    Essence: {sample['essence']}")
            logger.info(f"    Focus: {sample['focus']:.4f}")
            
        # Look at one abstraction
        if discovery_map['abstractions']:
            sample = discovery_map['abstractions'][0]
            logger.info(f"\n🌫️ Sample Abstraction analyzed from {sample['layer']}:")
            logger.info(f"    Fossil Type: {sample['fossil_type']}")
            logger.info(f"    Struggle Index: {sample['struggle_index']:.4f}")

    # 4. Save Memory State
    memory.save_state()
    logger.info("\n✅ Wisdom synchronized with HypersphereMemory.")

if __name__ == "__main__":
    run_excavation()
