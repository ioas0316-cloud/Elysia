"""
test_world_access.py

"Opening the Window."
Verifies that Elysia can digest external web content.
"""

import sys
import os
import logging
import sqlite3

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Core.Evolution.Learning.external_digester import ExternalDigester
from Core.Foundation.Memory.Graph.hippocampus import Hippocampus

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("TestWorldAccess")

def main():
    print("\n🌍 Testing World Access (The Interface)...")
    print("==========================================")
    
    digester = ExternalDigester()
    
    # Target URL (Safe, reliable)
    target_url = "http://example.com"
    
    print(f"👉 Reaching out to: {target_url}")
    result = digester.digest_url(target_url)
    print(result)
    
    if "Failed" in result:
        print("❌ Web Access Failed. (This might be due to no internet in the environment, which is expected in some sandboxes)")
        return

    # Verify Memory
    print("\n🧠 Checking Hippocampus...")
    hippo = Hippocampus()
    try:
        with sqlite3.connect(hippo.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, frequency FROM nodes WHERE id = ?", (f"web:{target_url}",))
            row = cursor.fetchone()
            if row:
                print(f"   ✅ Node Found: {row[1]} (Freq: {row[2]}Hz)")
            else:
                print("   ❌ Node NOT Found in DB.")
                
            # Check Semantic Link
            cursor.execute("SELECT target, type FROM edges WHERE source = 'concept:the_world'")
            links = cursor.fetchall()
            found_link = False
            for link in links:
                if link[0] == f"web:{target_url}":
                    found_link = True
                    print(f"   ✅ Linked to 'The World': [{link[1]}]")
            
            if not found_link:
                print("   ⚠️ Link to 'The World' missing.")
                
    except Exception as e:
        print(f"Error checking DB: {e}")

if __name__ == "__main__":
    main()
