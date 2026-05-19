
import sqlite3
import os

DB_PATH = "c:/Elysia/memory.db"

def inspect_db():
    if not os.path.exists(DB_PATH):
        print("❌ memory.db not found.")
        return

    print(f"📦 Inspecting {DB_PATH}...")
    file_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"📊 Size: {file_size_mb:.2f} MB")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"\n📑 Found {len(tables)} tables:")
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  - {table_name}: {count} rows")
            
            # Show schema sample
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [info[1] for info in cursor.fetchall()]
            print(f"    Columns: {', '.join(columns)}")

        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    inspect_db()
