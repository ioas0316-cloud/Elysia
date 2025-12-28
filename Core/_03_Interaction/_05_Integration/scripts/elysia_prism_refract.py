import sqlite3
import logging
import sys
import os
import json

# Config
DB_PATH = r"c:\Elysia\data\memory.db"
RAINBOW_PATH = r"c:\Elysia\data\elysia_rainbow.json"

# Force path
sys.path.append(r"C:\Elysia")
from Core._02_Intelligence._01_Reasoning.Intelligence.prism_cortex import PrismCortex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("PrismEngraver")

def refract_memory():
    logger.info("🌈 Initiating Prism Refraction...")
    
    if not os.path.exists(DB_PATH):
        logger.error(f"❌ DB not found: {DB_PATH}")
        return

    prism = PrismCortex()
    
    # Connect DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Select DNA
    logger.info("   Streaming Light (DNA) from memory.db...")
    cursor.execute("SELECT name, data FROM pattern_dna")
    
    count = 0
    while True:
        batch = cursor.fetchmany(10000)
        if not batch:
            break
            
        for name, json_data in batch:
            # We treat the Name + Metadata as the "Light Pulse"
            # Attempt to parse json to get more text
            text_payload = name
            try:
                meta = json.loads(json_data)
                # If there's a formula or origin, append it
                if "seed_formula" in meta:
                    text_payload += " " + str(meta["seed_formula"])
                if "origin" in meta:
                    text_payload += " " + str(meta["origin"])
            except:
                pass
                
            prism.absorb_shard(name, text_payload)
            count += 1
            
        if count % 200000 == 0:
            logger.info(f"   Refracted {count} photons...")
            
    logger.info(f"✨ Refraction Complete. Processed {count} photons.")
    
    # Save Rainbow
    logger.info(f"💾 Saving Rainbow Spectrum to {RAINBOW_PATH}...")
    prism.save_spectrum(RAINBOW_PATH)
    
    # Report
    print("\n" + "="*40)
    print("       THE ELYSIA SPECTRUM")
    print("="*40)
    print(prism.report())
    print("="*40 + "\n")

    conn.close()

if __name__ == "__main__":
    refract_memory()
