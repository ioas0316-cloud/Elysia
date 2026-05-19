import sqlite3
import zlib
import sys
import logging

# Config
DB_PATH = r"c:\Elysia\data\memory.db"

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("System")

def restore_concept(concept_name: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # 1. Fetch DNA
        cursor.execute("SELECT content_dna, data FROM pattern_dna WHERE name=?", (concept_name,))
        row = cursor.fetchone()
        
        if not row:
            logger.error(f"❌ Concept '{concept_name}' not found in DNA archive.")
            return

        content_dna, meta_data = row
        
        # 2. Decompress (Restoration)
        if content_dna:
            logger.info(f"🧬 DNA Found for '{concept_name}'. Resonating...")
            try:
                # Decompress zlib bytes
                restored_text = zlib.decompress(content_dna)
                # content_dna in concepts was a blob. 
                # If it was text originally, we assume utf-8.
                # However, the previous sample output showed `b'x\x9c...'` which is zlib header.
                # So we simply decompress.
                
                # Check if it needs decoding
                if isinstance(restored_text, bytes):
                    try:
                        decoded = restored_text.decode('utf-8')
                    except UnicodeDecodeError:
                        decoded = f"<Binary Data: {len(restored_text)} bytes>"
                else:
                    decoded = restored_text
                    
                print("\n" + "="*40)
                print(f"📖 RESTORED CONCEPT: {concept_name}")
                print("="*40)
                print(decoded)
                print("="*40 + "\n")
                
            except Exception as e:
                logger.error(f"⚠️ Decompression Failed: {e}")
                # Fallback to metadata
                logger.info(f"   Metadata: {meta_data}")
        else:
            logger.warning(f"⚠️ No 'content_dna' found (only metadata).")
            logger.info(f"   Metadata: {meta_data}")
            
    except Exception as e:
        logger.error(f"❌ Restoration Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python elysia_memory_restore.py <concept_name>")
        print("Example: python elysia_memory_restore.py genesis")
    else:
        restore_concept(sys.argv[1])
