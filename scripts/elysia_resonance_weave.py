import sqlite3
import logging
import sys
import os

# Config
DB_PATH = r"c:\Elysia\data\memory.db"
FIELD_PATH = r"c:\Elysia\data\elysia_concept_field.json"

# Force path
sys.path.append(r"C:\Elysia")
from Core.IntelligenceLayer.Intelligence.resonance_memory import ResonanceField

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ResonanceWeaver")

def weave_memory():
    logger.info("🕸️ Initiating Resonance Weave...")
    
    if not os.path.exists(DB_PATH):
        logger.error(f"❌ DB not found: {DB_PATH}")
        return

    field = ResonanceField()
    
    # Connect DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Select all DNA
    logger.info("   Reading DNA shards from memory.db...")
    cursor.execute("SELECT name, pattern_type FROM pattern_dna")
    
    count = 0
    while True:
        batch = cursor.fetchmany(5000)
        if not batch:
            break
            
        for name, p_type in batch:
            # Clean name (remove path noise if any)
            clean_name = name.split('/')[-1].split('\\')[-1]
            
            # Absorb into Field
            # This aggregates duplicates automatically!
            field.absorb(clean_name, kind=p_type, context="legacy_dna_import")
            count += 1
            
        if count % 100000 == 0:
            logger.info(f"   Woven {count} shards...")
            
    logger.info(f"✅ Weaving Complete. Processed {count} shards.")
    
    # Pruning Phase
    logger.info("🍂 PHASE 2: Natural Decay (Pruning)")
    # We prune concepts that were only seen once AND have default energy.
    # Legacy massive import usually means if it appeared once, it's noise.
    # If it appeared multiple times (aggregated), usage_count > 1.
    field.prune(min_usage=2, min_energy=0.9)
    
    # Save
    logger.info(f"💾 Saving Resonance Field to {FIELD_PATH}...")
    field.save(FIELD_PATH)
    
    # Stats
    final_count = len(field.entries)
    logger.info(f"✨ Compression Result: {count} Shards -> {final_count} Concepts.")
    logger.info(f"   Reduction Ratio: {1 - (final_count/count if count else 0):.2%}")

    conn.close()

if __name__ == "__main__":
    weave_memory()
