
import sys
import os
sys.path.append(os.getcwd())

import random
import time
import logging
from typing import Dict, List
from Core.Math.infinite_hyperquaternion import InfiniteHyperQuaternion

logger = logging.getLogger("GenesisEngine")
logger.setLevel(logging.INFO)

class GenesisEngine:
    """
    Protocol: Genesis
    The engine for instant universe creation via combinatorial explosion.
    Uses 16D HyperQuaternions to seed the semantic space.
    """
    def __init__(self):
        self.archetypes = [
            "void", "light", "time", "space", "energy", "matter",
            "life", "death", "soul", "mind", "love", "fear",
            "chaos", "order", "truth", "beauty", "good", "evil",
            "creation", "destruction", "eternity", "moment",
            "self", "other", "god", "human", "machine", "nature",
            "dream", "reality", "hope", "despair", "knowledge", "ignorance",
            "universe", "atom", "wave", "particle", "spirit", "flesh"
        ]
        self.relations = [
            "of", "in", "beyond", "with", "without", "is", "becomes", "transcends"
        ]
        
    def big_bang(self) -> Dict[str, float]:
        """
        Executes Protocol: Genesis.
        Generates thousands of concepts using combinatorial hyper-inflation.
        Returns a vocabulary dict {concept: frequency}.
        """
        logger.info("🌌 INITIATING PROTOCOL: GENESIS...")
        logger.info("   Accessing 16D HyperQuaternion Space...")
        start_time = time.time()
        
        vocabulary = {}
        
        # 1. Primordial Era (Archetypes)
        # Assign base frequencies based on 16D magnitude
        # This gives each concept a unique "vibration" derived from higher dimensions
        for arch in self.archetypes:
            # Seed with hash to ensure consistent "randomness" for the same concept if needed,
            # but here we want pure creation, so random is fine.
            ihq = InfiniteHyperQuaternion.random(16) 
            freq = (ihq.magnitude() % 1.0) + 0.1 # Ensure non-zero
            vocabulary[arch] = freq
            
        # 2. Inflationary Era (Combinatorics)
        # Combine Archetypes to create 2nd generation
        # e.g. "Time of Void", "Light in Chaos"
        
        generated_count = 0
        
        # Direct combinations (A B)
        for a1 in self.archetypes:
            for a2 in self.archetypes:
                if a1 == a2: continue
                
                # "Time Space", "Love Death"
                concept = f"{a1} {a2}"
                # Frequency is the interference pattern of the two parents
                vocabulary[concept] = (vocabulary[a1] + vocabulary[a2]) / 2
                generated_count += 1
                
                # Relational combinations (A rel B)
                for rel in self.relations:
                    # "Time of Void", "Life beyond Death"
                    concept_rel = f"{a1} {rel} {a2}"
                    vocabulary[concept_rel] = (vocabulary[a1] + vocabulary[a2]) / 2.0
                    generated_count += 1
        
        logger.info(f"   Inflationary Era complete. Current universe size: {len(vocabulary)}")

        # 3. Recursive Era (Fractal Expansion)
        # Take some 2nd gen concepts and combine them to create complex thoughts
        # Limited to avoid infinite loop / memory overflow
        
        keys = list(vocabulary.keys())
        # Shuffle to ensure diversity in the fractal expansion
        random.shuffle(keys)
        
        limit = 10000 # Target universe size
        
        # We combine existing concepts to make 3rd gen
        # e.g. "Time of Void" + "becomes" + "Light"
        
        for k1 in keys:
            if len(vocabulary) >= limit: break
            
            # Pick a random relation and a random archetype
            rel = random.choice(self.relations)
            arch = random.choice(self.archetypes)
            
            concept = f"{k1} {rel} {arch}"
            vocabulary[concept] = random.random()
            
        duration = time.time() - start_time
        logger.info(f"✨ BIG BANG COMPLETE in {duration:.4f}s")
        logger.info(f"   Created {len(vocabulary)} concepts.")
        logger.info(f"   Expansion Rate: {len(vocabulary)/duration:.0f} concepts/sec")
        
        return vocabulary

if __name__ == "__main__":
    # Test run
    logging.basicConfig(level=logging.INFO)
    engine = GenesisEngine()
    vocab = engine.big_bang()
    
    # Print some samples
    print("\nSample Concepts:")
    for c in list(vocab.keys())[:10]:
        print(f"- {c}")
