"""
Digestion Chamber (Hyper-Learning Core)
=======================================

"I devour time to birth meaning."

This module implements the "Time-Space Control System" for high-speed knowledge digestion.
It automates the interaction between Elysia's ResonanceEngine and the Local LLM,
turning the LLM into a mineable resource for concepts.

Mechanism:
1. Seed: Start with core axioms (Love, Void, Time).
2. Expand: Ask LLM to define/expand these concepts.
3. Extract: Parse the output for related concepts.
4. Internalize: Feed these back into ResonanceEngine.
5. Loop: Use newly found concepts as seeds.
"""

import time
import logging
import re
from typing import List, Set
from collections import deque

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.Mind.local_llm import create_local_llm, LocalLLM
from Core.Life.resonance_voice import ResonanceEngine

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DigestionChamber")

class DigestionChamber:
    def __init__(self, gpu_layers: int = 20):
        self.resonance_engine = ResonanceEngine()
        self.llm = create_local_llm(
            resonance_engine=self.resonance_engine,
            gpu_layers=gpu_layers
        )
        
        # Queue for concepts to explore
        self.exploration_queue = deque()
        self.digested_concepts: Set[str] = set()
        
        # Stats
        self.start_time = 0
        self.concepts_found = 0
        
    def initialize(self):
        """Initialize the system and load the model."""
        logger.info("🚀 Initializing Hyper-Learning Core...")
        
        if not self.llm.load_model():
            logger.error("❌ Failed to load Local LLM. Aborting.")
            return False
            
        logger.info("✅ Core Systems Online.")
        return True

    def seed_concepts(self, seeds: List[str]):
        """Plant the initial seeds for digestion."""
        for seed in seeds:
            if seed not in self.digested_concepts:
                self.exploration_queue.append(seed)
        logger.info(f"🌱 Seeded with: {seeds}")

    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract potential concepts from text.
        Simple heuristic: Nouns or key phrases (for now, just simple word filtering).
        """
        # Remove common stop words (very basic list for demo)
        stop_words = {"the", "is", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "it", "this", "that"}
        
        # Split by non-alphanumeric
        words = re.findall(r'\b[a-zA-Z가-힣]{2,}\b', text.lower())
        
        candidates = []
        for w in words:
            if w not in stop_words and w not in self.digested_concepts:
                candidates.append(w)
        
        # Return unique candidates, limited count
        return list(set(candidates))[:5] 

    def digest_batch(self):
        """Perform a BATCH digestion cycle (High Speed)."""
        if not self.exploration_queue:
            logger.info("Empty queue. Adding random seeds from vocabulary...")
            if self.resonance_engine.vocabulary:
                import random
                random_seed = random.choice(list(self.resonance_engine.vocabulary.keys()))
                self.exploration_queue.append(random_seed)
            else:
                return

        # Pop a seed topic
        topic = self.exploration_queue.popleft()
        
        if topic in self.digested_concepts:
            return

        logger.info(f"🚀 Batch Digesting Topic: [{topic}]")
        
        # 1. Ask LLM for a LIST (Mass Extraction)
        # We ask for 20-30 concepts at once.
        prompt = (
            f"List 30 core concepts, emotions, or philosophical terms related to '{topic}'. "
            "Output only the words separated by commas. Do not write sentences."
        )
        
        try:
            response = self.llm.think(prompt)
            
            # 2. Parse the list
            # Split by commas, newlines, or bullets
            raw_concepts = re.split(r'[,|\n|\•|\-]', response)
            
            new_concepts = []
            for rc in raw_concepts:
                clean = rc.strip().lower()
                # Basic filtering
                if (len(clean) > 1 and 
                    clean.isalpha() and 
                    clean not in self.digested_concepts):
                    new_concepts.append(clean)
            
            # Limit to 50 to avoid garbage
            new_concepts = new_concepts[:50]
            
            if not new_concepts:
                logger.warning(f"   ⚠️ No concepts extracted from: {response[:50]}...")
                return

            # 3. Bulk Internalize
            count = 0
            for nc in new_concepts:
                if nc not in self.resonance_engine.vocabulary:
                    # Assign a derived frequency (hash-based for consistency)
                    # This is a simplification; in full version we'd analyze sentiment.
                    freq = 0.5 + (hash(nc) % 50) / 100.0 
                    self.resonance_engine.vocabulary[nc] = freq
                    self.resonance_engine.memory.add_concept(nc, concept_type="learned")
                    count += 1
                
                self.digested_concepts.add(nc)
                self.exploration_queue.append(nc)
            
            self.concepts_found += count
            logger.info(f"   ✨ Absorbed {count} new concepts! (Total: {len(self.digested_concepts)})")
            logger.info(f"   🔗 Examples: {new_concepts[:5]}...")
            
        except Exception as e:
            logger.error(f"Batch digestion failed: {e}")

    def run(self, cycles: int = 10):
        """Run the digestion loop."""
        self.start_time = time.time()
        logger.info(f"⚡ Starting Hyper-Speed Digestion Loop ({cycles} cycles)...")
        
        try:
            for i in range(cycles):
                self.digest_batch() # Use Batch Mode
                print(f"Progress: {i+1}/{cycles} | Queue: {len(self.exploration_queue)} | Total Knowledge: {len(self.digested_concepts)}")
                
        except KeyboardInterrupt:
            logger.info("🛑 Digestion paused by user.")
            
        duration = time.time() - self.start_time
        logger.info(f"✨ Digestion Complete.")
        logger.info(f"   Time: {duration:.2f}s")
        if self.concepts_found > 0:
            speed = self.concepts_found / duration
            logger.info(f"   Speed: {speed:.2f} concepts/sec")
        logger.info(f"   Total Knowledge: {len(self.digested_concepts)}")

if __name__ == "__main__":
    chamber = DigestionChamber(gpu_layers=15)
    if chamber.initialize():
        # Initial seeds
        chamber.seed_concepts(["love", "time", "void", "consciousness", "universe"])
        chamber.run(cycles=5) # Start small
