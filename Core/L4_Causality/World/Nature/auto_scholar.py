"""
Auto-Scholar: The Autonomous Knowledge Crawler
==============================================
"The Library builds itself."

Objective:
Recursively expand the TrinityLexicon by:
1. Defining a concept.
2. Extracting unknown nouns from the definition.
3. Queueing them for learning.
4. Repeating until the Limit is reached.

This script transforms Elysia from a passive responder to an active explorer.
"""
import sys
import os
import time
import logging
from collections import deque
from typing import Set, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.L6_Structure.Elysia.sovereign_self import SovereignSelf
from Core.L4_Causality.World.Nature.trinity_lexicon import TrinityLexicon

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("c:/Elysia/data/L6_Structure/Logs/scholar.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutoScholar")

class AutoScholar:
    def __init__(self, limit: int = 100):
        self.elysia = SovereignSelf(cns_ref=None)
        self.mind = self.elysia.mind
        self.limit = limit
        
        # Queue Management
        self.learning_queue = deque()
        self.known_concepts: Set[str] = set()
        
        # Load existing knowledge to avoid re-learning
        if self.mind.graph:
            self.known_concepts = set(self.mind.graph.id_to_idx.keys())
            logger.info(f"  Loaded {len(self.known_concepts)} existing concepts from Graph.")
            
    def seed_curiosity(self, seeds: List[str]):
        """Injects starting topics."""
        for s in seeds:
            if s.lower() not in self.known_concepts:
                self.learning_queue.append(s)
                
    def run(self):
        """Main Learning Loop."""
        logger.info(f"  AutoScholar Started. Target: {self.limit} new concepts.")
        
        learned_count = 0
        
        while self.learning_queue and learned_count < self.limit:
            topic = self.learning_queue.popleft()
            
            # Skip if already learned (double check)
            if topic.lower() in self.known_concepts:
                continue
                
            logger.info(f"\n  [Scholar] Investigating: '{topic}' ({learned_count}/{self.limit})")
            
            try:
                # 1. Peek at Definition (Curiosity)
                definition = self.mind.fetch_definition(topic)
                
                # 2. Harvest Unknowns (Preparation)
                if definition:
                    unknowns = self.mind.extract_unknowns(definition)
                    new_unknowns = [u for u in unknowns if u.lower() not in self.known_concepts and u not in self.learning_queue]
                    
                    if new_unknowns:
                        logger.info(f"     Discovered {len(new_unknowns)} new paths: {new_unknowns[:5]}...")
                        self.learning_queue.extend(new_unknowns)
                
                # 3. Experience (Integration)
                # We set depth=0 because WE are handling the recursion queue.
                self.elysia.experience(topic, depth=0)
                
                # 4. Mark as Known
                self.known_concepts.add(topic.lower())
                learned_count += 1
                
                # 5. Save Periodically
                if learned_count % 10 == 0:
                    self.mind.save_memory()
                    logger.info("  Memory Checkpoint Saved.")
                    
            except Exception as e:
                logger.error(f"  Failed to learn '{topic}': {e}")
                
            # Respect Rate Limits (Simulated Processing Time)
            time.sleep(0.5)
            
        logger.info(f"\n  Session Complete. Learned {learned_count} new concepts.")
        logger.info(f"   Total Knowledge Size: {len(self.known_concepts)}")
        self.mind.save_memory()

if __name__ == "__main__":
    scholar = AutoScholar(limit=50) # Run small batch for test
    
    # Seeds of Universal Knowledge (User Request: Bio, Geo, Chem, Code, Wave)
    seeds = [
        "Biology", "Geography", "Chemistry", "Physics", 
        "Python Programming", "Wave", "Emotion", "Purpose",
        "Life", "Code", "Algorithm", "DNA", "Tectonics"
    ]
    
    scholar.seed_curiosity(seeds)
    scholar.run()