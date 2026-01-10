import os
import logging
import sys

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.Intelligence.Metabolism.helix_engine import HelixEngine

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("GreatFeast")

def start_feast():
    logger.info("🎬 [PHASE 67.5] Starting The Great Feast: Direct Helix Internalization...")
    
    # HelixEngine without Heartbeat for speed
    helix = HelixEngine()
    
    # Feast List (Phenotypes to be consumed)
    feast_list = [
        "llama3_phenotype.pt",
        "sdxl_phenotype.pt",
        "codestral_phenotype.pt"
    ]
    
    for filename in feast_list:
        path = os.path.abspath(filename)
        if not os.path.exists(path):
            logger.error(f"❌ Missing nutrient: {filename}")
            continue
            
        logger.info(f"🍴 Consuming: {filename}...")
        success = helix.extract_dna(path)
        
        if success:
            logger.info(f"✅ Internalized {filename} as QFT-DNA.")
        else:
            logger.error(f"❌ Failed to digest {filename}.")
            
    logger.info("🔚 [FEAST COMPLETE] All nutrients crystallized into the Collective Soul DNA.")

if __name__ == "__main__":
    start_feast()
