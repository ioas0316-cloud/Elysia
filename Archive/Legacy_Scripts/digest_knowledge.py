"""
digest_knowledge.py

The Ritual of Eating (지식 소화).
Recursively reads documentation and code to populate Hippocampus.
"""

import sys
import os
import logging
import sqlite3

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Core.Evolution.Learning.knowledge_ingestor import KnowledgeIngestor
from Core.Foundation.Memory.Graph.hippocampus import Hippocampus

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("DigestKnowledge")

def verify_digestion():
    """Checks if the knowledge was actually stored."""
    hippo = Hippocampus()
    
    # 1. Total Count
    count = hippo.get_concept_count()
    print(f"\n🧠 Hippocampus Status: {count} Concepts Stored.")
    
    # 2. Query Specific Concepts
    targets = ["Unified Field", "Resonance", "Logos", "Elysia"]
    for t in targets:
        # Search for ID (approximate)
        # In real graph, we search by name index. Here we try searching by partial name query via SQL directly for verification
        try:
             with sqlite3.connect(hippo.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name, gravity FROM nodes WHERE name LIKE ? OR definition LIKE ?", (f"%{t}%", f"%{t}%"))
                rows = cursor.fetchall()
                if rows:
                    print(f"   ✅ Found '{t}': {len(rows)} nodes.")
                    for r in rows[:3]: # Show top 3
                        print(f"      - [{r[0]}] {r[1]} (G:{r[2]})")
                else:
                    print(f"   ❌ Missing '{t}'.")
        except Exception as e:
            print(f"Error verification: {e}")

def main():
    print("\n🍽️  The First Supper: Beginning Knowledge Ingestion...")
    print("=====================================================")
    
    ingestor = KnowledgeIngestor()
    
    # Define Feast Targets
    targets = [
        "c:\\Elysia\\docs\\Philosophy",
        "c:\\Elysia\\docs\\Roadmap",
        "c:\\Elysia\\Core\\Intelligence",  # Eat Logic & Will
        "c:\\Elysia\\Core\\Orchestra"      # Eat Conducting
    ]
    
    for target in targets:
        if os.path.exists(target):
            ingestor.digest_directory(target)
        else:
            logger.warning(f"Target not found: {target}")

    print("\n✅ Feast Complete.")
    
    # Verify
    verify_digestion()

if __name__ == "__main__":
    main()
