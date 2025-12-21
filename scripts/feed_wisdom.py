"""
Script: Feed Wisdom (Use Existing Digestion System)
===================================================

"Do not build a new organ. Use the stomach that already exists."

This script uses the EXISTING `Core/Foundation/knowledge_acquisition.py`
to ingest the Literature Corpus. It does NOT create new classes.
"""

import sys
import os
import glob
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.knowledge_acquisition import KnowledgeAcquisitionSystem
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeedWisdom")

def feed_elysia():
    print("🍽️  Feeding Elysia with Literature...")
    print("===================================")
    
    # 1. Get the Stomach (Existing System)
    stomach = KnowledgeAcquisitionSystem()
    
    # 2. Find Food (Data)
    # Focusing on high-quality literature as requested
    corpus_path = Path("c:/Elysia/data/corpus/literature")
    files = list(corpus_path.glob("*.txt"))
    
    if not files:
        # Fallback if no specific literature files found, try drama
        corpus_path = Path("c:/Elysia/data")
        files = list(corpus_path.glob("drama_*.txt"))
    
    print(f"found {len(files)} items of 'Proper Knowledge Data'")
    
    # 3. Digestion Cycle
    curriculum = []
    
    for file_path in files[:3]: # Limit to 3 for demonstration speed
        try:
            content = file_path.read_text(encoding='utf-8')
            # Extract title from filename
            title = file_path.stem.replace("drama_", "").replace("_", " ").title()
            
            # Create a "Nutrient Packet"
            nutrient = {
                "concept": title, # The Concept Name
                "description": content[:1000] + "..." # First 1000 chars as essence
            }
            curriculum.append(nutrient)
            print(f"   - Prepared: {title}")
        except Exception as e:
            logger.error(f"Failed to prep {file_path}: {e}")
            
    # 4. Active Eating
    if curriculum:
        print("\n😋 Eating...")
        stomach.learn_curriculum(curriculum)
        
        print("\n✨ Digestion Complete.")
        print("   The knowledge is now part of the Internal Universe.")
        
        # Verify Absorption
        stats = stomach.get_knowledge_stats()
        print(f"   Total Knowledge Nodes: {stats['concepts_in_universe']}")
    else:
        print("❌ No food found!")

if __name__ == "__main__":
    feed_elysia()
