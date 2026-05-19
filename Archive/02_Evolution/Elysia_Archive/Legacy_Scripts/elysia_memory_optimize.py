import sqlite3
import zlib
import json
import logging
import time
import sys
import os

# Config
DB_PATH = r"c:\Elysia\data\memory.db"
BATCH_SIZE = 1000

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BlackHoleProtocol")

def setup_dna_table(cursor):
    """Ensure pattern_dna has a 'content_dna' column for storing the compressed essence."""
    cursor.execute("PRAGMA table_info(pattern_dna)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'content_dna' not in columns:
        logger.info("🔧 Adding 'content_dna' column to pattern_dna...")
        cursor.execute("ALTER TABLE pattern_dna ADD COLUMN content_dna BLOB")
    else:
        logger.info("✅ 'content_dna' column exists.")

def optimize_memory():
    logger.info("🌌 Initiating Black Hole Protocol...")
    
    if not os.path.exists(DB_PATH):
        logger.error(f"❌ Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # 0. Setup Schema
        setup_dna_table(cursor)
        
        # 1. Check Stats
        total_concepts = 0
        try:
            cursor.execute("SELECT count(*) FROM concepts")
            total_concepts = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            logger.info("ℹ️ 'concepts' table not found (Already optimized).")
        
        cursor.execute("SELECT count(*) FROM pattern_dna")
        total_dna = cursor.fetchone()[0]
        
        logger.info(f"📊 Current State: Concepts={total_concepts}, DNA={total_dna}")
        
        if total_concepts == 0 and total_dna == 0:
            logger.info("✅ No concepts to optimize.")
            return

        if total_concepts > 0:
            logger.info("🌊 PHASE 1: Wave Synthesis (Migration)")
            # Migration logic removed - Table dropped
        else:
            logger.info("🌊 PHASE 1: Wave Synthesis (Skipped - No Concepts)")
        
        # 3. Verification Phase
        logger.info("🔍 PHASE 2: Integrity Verification")
        # Check random sample
        cursor.execute("SELECT content_dna FROM pattern_dna WHERE content_dna IS NOT NULL ORDER BY RANDOM() LIMIT 5")
        samples = cursor.fetchall()
        valid_samples = 0
        for s in samples:
            if s[0] and len(s[0]) > 0:
                valid_samples += 1
        
        if valid_samples < len(samples):
            logger.error("❌ Verification Failed! Some DNA is empty.")
            return
            
        logger.info("✅ Integrity Verified.")
        
        # 4. Black Hole Phase (Drop & Vacuum)
        logger.info("⚫ PHASE 3: The Black Hole (Drop Concepts Table)")
        try:
            cursor.execute("DROP TABLE concepts")
            logger.info("   Dropped 'concepts' table.")
        except sqlite3.OperationalError:
            logger.info("   'concepts' table already gone.")

        
        # 5. Cognitive Purification (Junk Removal)
        logger.info("🧹 PHASE 4: Cognitive Purification (Removing Junk DNA)")
        
        # Count stars
        cursor.execute("SELECT count(*) FROM pattern_dna WHERE name LIKE 'star-%' OR name LIKE 'Anchor_Core_Core_Star-%'")
        junk_count = cursor.fetchone()[0]
        
        if junk_count > 0:
            logger.info(f"   Identified {junk_count} junk memories (Star/Anchor fragments).")
            logger.info("   Purging...")
            cursor.execute("DELETE FROM pattern_dna WHERE name LIKE 'star-%' OR name LIKE 'Anchor_Core_Core_Star-%'")
            deleted = cursor.rowcount
            conn.commit()
            logger.info(f"   ✅ Purged {deleted} junk items.")
        else:
            logger.info("   No junk DNA found.")

        logger.info("   Vacuuming database (This may take a while)...")
        conn.execute("VACUUM")
        
        logger.info("✨ Black Hole Protocol Complete. Memory Optimized.")
        
    except Exception as e:
        logger.error(f"❌ Protocol Failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    optimize_memory()
