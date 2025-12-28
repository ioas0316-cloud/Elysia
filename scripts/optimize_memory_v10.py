
import sqlite3
import os
import time
from pathlib import Path

DB_PATH = Path("c:/Elysia/memory.db")
MAX_CACHE_AGE_DAYS = 30

def optimize_db():
    print("🧹 Optimizing Elysia Memory (v10.0)...")
    
    if not DB_PATH.exists():
        print("❌ memory.db not found.")
        return

    orig_size = os.path.getsize(DB_PATH)
    print(f"  - Initial Size: {orig_size / 1024:.2f} KB")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 1. Prune Old Sensory Cache
        # (Assuming we have data, which we might not yet)
        try:
            print("  - Pruning old sensory cache...")
            cursor.execute("DELETE FROM sensory_cache WHERE last_accessed < date('now', '-30 days')")
            deleted_rows = cursor.rowcount
            print(f"    Deleted {deleted_rows} old cache entries.")
        except sqlite3.OperationalError:
            print("    (sensory_cache table might not exist yet)")

        # 2. Vacuum
        print("  - Running VACUUM...")
        conn.commit()
        conn.execute("VACUUM")
        
        # 3. Analyze
        print("  - Running ANALYZE...")
        conn.execute("ANALYZE")
        
        conn.close()
        
        new_size = os.path.getsize(DB_PATH)
        print(f"  - Final Size: {new_size / 1024:.2f} KB")
        print(f"✨ Optimization Complete. Saved {(orig_size - new_size)/1024:.2f} KB")

    except Exception as e:
        print(f"❌ Error during optimization: {e}")

if __name__ == "__main__":
    optimize_db()
