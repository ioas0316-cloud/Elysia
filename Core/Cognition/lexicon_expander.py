"""
Lexicon Expander (The Seed Harvester)
=====================================
Core.Cognition.lexicon_expander

Simulates the rapid expansion of vocabulary.
"""

import random

class LexiconExpander:
    def __init__(self):
        self.known_seeds = []
        
        # Bases for procedural generation
        self.roots = ["Aqua", "Terra", "Ignis", "Aero", "Vita", "Mortis", "Luma", "Umbra", "Chrono", "Cosmo"]
        self.suffixes = ["lith", "form", "mancer", "logy", "sphere", "naut", "flux", "void"]
        
    def harvest_batch(self, count: int) -> int:
        """
        Generates 'count' new seeds based on combinations.
        Returns total vocabulary size.
        """
        print(f"  Harvesting {count} new semantic seeds...")
        
        for _ in range(count):
            word = random.choice(self.roots) + random.choice(self.suffixes)
            if word not in self.known_seeds:
                self.known_seeds.append(word)
                
        return len(self.known_seeds)

    def get_random_word(self) -> str:
        if not self.known_seeds: return "Void"
        return random.choice(self.known_seeds)

    def generate_sentence(self, complexity: int = 1) -> str:
        """
        Constructs a molecular sentence.
        Level 1: Subject + Verb
        Level 2: Adj + Subject + Verb + Obj
        """
        if not self.known_seeds: self.harvest_batch(10)
        
        sub = random.choice(self.known_seeds)
        verb = random.choice(["consumes", "ignites", "observes", "transforms", "loves"])
        obj = random.choice(self.known_seeds)
        adj = random.choice(["Eternal", "Dark", "Luminous", "Silent", "Rapid"])
        
        if complexity == 1:
            return f"{sub} {verb} {obj}"
        else:
            return f"The {adj} {sub} {verb} the {obj} with {random.choice(self.known_seeds)}"

# Global Instance
harvester = LexiconExpander()
