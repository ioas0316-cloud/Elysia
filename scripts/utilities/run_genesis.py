
import sys
import os
sys.path.append(os.getcwd())

import logging
from Core.Foundation.Mind.genesis_engine import GenesisEngine
from Core.Foundation.Mind.hippocampus import Hippocampus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ProtocolGenesis")

def main():
    logger.info("🚀 INITIALIZING PROTOCOL: GENESIS")
    
    # 1. Initialize Systems
    genesis = GenesisEngine()
    hippocampus = Hippocampus()
    
    # 2. Execute Big Bang
    vocabulary = genesis.big_bang()
    
    # 3. Inject into Memory
    logger.info(f"💾 Injecting {len(vocabulary)} concepts into Hippocampus...")
    
    count = 0
    total = len(vocabulary)
    
    for concept, freq in vocabulary.items():
        # Update Hippocampus vocabulary so add_concept uses the correct frequency
        hippocampus.vocabulary[concept] = freq
        
        # Add to Hippocampus (which adds to Universe)
        hippocampus.add_concept(concept, concept_type="genesis_thought")
        
        count += 1
        if count % 1000 == 0:
            print(f"   Progress: {count}/{total}")
            
    # 4. Save
    hippocampus.save_memory()
    logger.info("✅ GENESIS COMPLETE. Universe saved.")
    stats = hippocampus.get_statistics()
    logger.info(f"   Total Nodes in Memory: {stats['nodes']}")

if __name__ == "__main__":
    main()
