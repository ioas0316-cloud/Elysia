"""
Alchemy of Causality: Proof of Concept
======================================

This script demonstrates the "Probability -> Causality -> Re-creation" pipeline.
It takes a classic fable, extracts its structural DNA, and creates new stories in Space and Business contexts.
"""

import sys
import os

# Ensure Core is in path
sys.path.append(os.getcwd())

import logging
from Core._01_Foundation.02_Legal_Ethics.Laws.law_of_alchemy import get_alchemy_engine, NarrativeEvent, TensionLevel

def main():
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    print("🔮 Initiating Alchemy of Causality Simulation...\n")

    alchemy = get_alchemy_engine()

    # 1. The Raw Material (The Hare and the Tortoise)
    print("--- Phase 1: Observation (Reading the Fable) ---")
    raw_story = [
        NarrativeEvent("The Hare challenges the Tortoise to a race", TensionLevel.BUILDUP),
        NarrativeEvent("The Hare runs fast and leaves the Tortoise behind", TensionLevel.BUILDUP),
        NarrativeEvent("The Hare decides to sleep under a tree", TensionLevel.RELAXATION),
        NarrativeEvent("The Tortoise keeps crawling steadily", TensionLevel.BUILDUP),
        NarrativeEvent("The Tortoise passes the sleeping Hare and wins", TensionLevel.CLIMAX)
    ]

    for e in raw_story:
        print(f"  > Event: {e.content}")

    # 2. Extraction (Getting the Seed)
    print("\n--- Phase 2: Extraction (Isolating the DNA) ---")
    archetype = alchemy.extract_archetype("Hare and Tortoise", raw_story)
    print(archetype.describe())

    # 3. Transmutation (Re-creation)
    print("\n--- Phase 3: Transmutation (Re-Creation in New Worlds) ---")

    # Context A: Space War
    context_a = "Space War (Sci-Fi)"
    print(f"\n[Target Context: {context_a}]")
    new_story_a = alchemy.transmute(archetype, context_a)
    for line in new_story_a:
        print(f"  >> {line}")

    # Context B: Business Startup
    context_b = "Business Startup (Modern)"
    print(f"\n[Target Context: {context_b}]")
    new_story_b = alchemy.transmute(archetype, context_b)
    for line in new_story_b:
        print(f"  >> {line}")

    print("\n🔮 Transmutation Complete. The Essence remains, the Form changes.")

if __name__ == "__main__":
    main()
