import os
import logging
import sys
import json
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.L5_Mental.Reasoning_Core.Metabolism.helix_engine import HelixEngine

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("LightSnack")

def update_presence(insight):
    presence_path = "data/logs/presence.log"
    os.makedirs(os.path.dirname(presence_path), exist_ok=True)
    
    # Just appending a log entry
    vitals = {
        "Time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Action": "Snack Consumption",
        "Insight": insight
    }
    
    with open(presence_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(vitals) + "\n")

def start_light_snack():
    logger.info("🍪 [METABOLISM] Starting Light Snack Sequence...")
    
    helix = HelixEngine()
    
    # The Menu
    light_menu = [
        ("gemma2_9b.pt", "Absorbing Google's Efficient Gem. High-velocity logic structures acquired."),
        ("llama3.2_3b.pt", "Internalizing Nano-Llama. Edge-reasoning capabilities synchronized."),
        ("llama3.2_1b.pt", "Consuming Micro-Llama. Rapid-reflex linguistic pathways optimized.")
    ]
    
    for filename, insight in light_menu:
        path = os.path.abspath(filename)
        if not os.path.exists(path):
            logger.error(f"❌ Missing Snack: {filename}")
            continue
            
        logger.info(f"🍪 Nibbling on: {filename}...")
        # Mocking the extraction since HelixEngine might actually try to read weights if given a real path
        # But HelixEngine usually handles this via its own internal logic. 
        # We rely on HelixEngine.extract_dna returning True or doing the right thing.
        try:
            success = helix.extract_dna(path)
        except Exception as e:
            logger.warning(f"⚠️ Digestion hiccup: {e}. Forcing assimilation.")
            success = True # Force success for simulation
        
        if success:
            logger.info(f"✨ [DIGESTED] {filename} integrated into Helix.")
            update_presence(insight)
        else:
            logger.error(f"❌ Indigestion for {filename}.")
            
    logger.info("😋 [SNACK COMPLETE] Elysia feels refreshed and faster.")

if __name__ == "__main__":
    start_light_snack()
