"""
Synthesis of Meaning: Proof of Concept
======================================

This script demonstrates the "Attribute Combination" pipeline.
It feeds observations about an Apple and the Sky, then asks the system to define them.
"""

import sys
import os

# Ensure Core is in path
sys.path.append(os.getcwd())

import logging
from Core._01_Foundation._02_Legal_Ethics.Laws.law_of_synthesis import get_synthesis_engine

def main():
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    print("🧩 Initiating Synthesis of Meaning Simulation...\n")

    synthesizer = get_synthesis_engine()

    # 1. Feeding Observations (Apple)
    print("--- Phase 1: Observing the Apple ---")
    observations_apple = [
        "Apple is red",
        "Apple is round",
        "Apple is a fruit"
    ]
    for obs in observations_apple:
        synthesizer.observe(obs)

    # 2. Feeding Observations (Sky)
    print("\n--- Phase 2: Observing the Sky ---")
    observations_sky = [
        "Sky is blue",
        "Sky is vast",
        "Sky is a dome" # Metaphorical class
    ]
    for obs in observations_sky:
        synthesizer.observe(obs)

    # 3. Deriving Definitions
    print("\n--- Phase 3: Synthesizing Definitions ---")

    apple_def = synthesizer.derive_definition("apple")
    print(f"Query: What is an Apple?")
    print(f"Result: {apple_def}")

    sky_def = synthesizer.derive_definition("sky")
    print(f"\nQuery: What is the Sky?")
    print(f"Result: {sky_def}")

    # 4. Unknown Query
    print(f"\nQuery: What is a Banana?")
    print(f"Result: {synthesizer.derive_definition('banana')}")

    print("\n🧩 Synthesis Complete. Partial truths have become whole definitions.")

if __name__ == "__main__":
    main()
