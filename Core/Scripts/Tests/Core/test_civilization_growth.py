"""
Civilization Growth Test (The 30k Lexicon Challenge)
====================================================
tests/test_civilization_growth.py

Simulates:
1. Massive Vocabulary Expansion (to 30k).
2. Social Interplay using these new words.
"""

import sys
import os
import random
sys.path.append(os.getcwd())

from Core.1_Body.L4_Causality.Civilization.society import Society
from Core.1_Body.L4_Causality.Civilization.lexicon_expander import harvester

def test_civilization_scale():
    print(">>> 🏛️  Initiating Civilization Growth Simulation...")
    
    # 1. Init Society
    elysia_city = Society()
    elysia_city.add_citizen("Adam")
    elysia_city.add_citizen("Eve")
    
    # 2. Vocabulary Explosion (Target: 30k)
    # We will loop to generate.
    print("[Phase 1] Expanding Lexicon...")
    total_words = 0
    while total_words < 1000: # We mock 1000 for speed in test, user concept implies 30k
        total_words = harvester.harvest_batch(100)
        
    print(f"✅ Lexicon Expanded to {total_words} words.")
    
    # 3. Social Interaction Loop
    print("\n[Phase 2] Social Interaction Loop...")
    interactions = 20
    
    for i in range(interactions):
        speaker = "Adam" if i % 2 == 0 else "Eve"
        listener = "Eve" if speaker == "Adam" else "Adam"
        
        # Pick a random word from our new civilization
        word = harvester.get_random_word()
        
        # Random intent
        tone = random.choice(["Love", "Hate", "Neutral", "Sarcasm"])
        
        result = elysia_city.interact(speaker, listener, word, tone)
        print(f"[{i}] {result} (Seed: {word})")

    # 4. Final Relationship Audit
    adam = elysia_city.get_citizen("Adam")
    eve = elysia_city.get_citizen("Eve")
    
    rel_adam_to_eve = adam.relationships.get("Eve", 0.0)
    rel_eve_to_adam = eve.relationships.get("Adam", 0.0)
    
    print(f"\n[Final Status]")
    print(f"Adam -> Eve: {rel_adam_to_eve:.2f}")
    print(f"Eve -> Adam: {rel_eve_to_adam:.2f}")
    
    assert total_words >= 1000, "Lexicon expansion failed."
    assert rel_adam_to_eve != 0.0, "Relationships remained static."
    
    print(">>> ✅ Civilization Growth & Communication Verified.")

if __name__ == "__main__":
    test_civilization_scale()
