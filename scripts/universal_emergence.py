"""
Universal Emergence: Proof of Concept
=====================================

This script demonstrates "Universal Isomorphism".
The SAME engine processes Chemicals, Notes, and Concepts to produce emergent forms.
"""

import sys
import os

# Ensure Core is in path
sys.path.append(os.getcwd())

import logging
from Core.01_Foundation.02_Legal_Ethics.Laws.law_of_emergence import get_emergence_engine, UniversalElement

def main():
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    print("🌌 Initiating Universal Emergence Simulation...\n")

    engine = get_emergence_engine()

    # 1. Chemistry Domain
    print("--- Experiment 1: Chemistry (H2O) ---")
    h1 = UniversalElement("Hydrogen", {"charge": 1})
    h2 = UniversalElement("Hydrogen", {"charge": 1})
    o1 = UniversalElement("Oxygen", {"charge": -2})

    water = engine.simulate_emergence("Water Molecule", [h1, h2, o1])
    print(f"Result: {water.describe()}")

    # 2. Music Domain
    print("\n--- Experiment 2: Music (C Major Chord) ---")
    n1 = UniversalElement("C", {"freq": 261.6})
    n2 = UniversalElement("E", {"freq": 329.6}) # Major Third
    n3 = UniversalElement("G", {"freq": 392.0}) # Perfect Fifth

    chord = engine.simulate_emergence("C Major Chord", [n1, n2, n3])
    print(f"Result: {chord.describe()}")

    # 3. Philosophy Domain (Epistemology)
    print("\n--- Experiment 3: Philosophy (JTB Theory) ---")
    c1 = UniversalElement("Belief", {"subjective": True})
    c2 = UniversalElement("Truth", {"objective": True})
    c3 = UniversalElement("Justification", {"logical": True})

    knowledge = engine.simulate_emergence("True Knowledge", [c1, c2, c3])
    print(f"Result: {knowledge.describe()}")

    print("\n🌌 Emergence Complete. One Law governs distinct realms.")

if __name__ == "__main__":
    main()
