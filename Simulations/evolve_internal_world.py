"""
Evolve Internal World (The Chronos Chamber)
===========================================
"To understand a century, populate it with seconds."

This script accelerates the Internal Universe to simulate 30 years of experience.
It generates thousands of 'Synthetic Memories' and plants them in the Hippocampus.

Goal: Provide the "Density of Experience" the user requested.
"""

import sys
import os
import logging

# Add Root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Core.Foundation.internal_universe import InternalUniverse

def evolve():
    # Setup Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Evolution")
    
    print("\n🌌 Spinnin up the Internal Universe...")
    universe = InternalUniverse()
    
    print("\n⏳ Entering the Chronos Chamber...")
    print("   Target: Simulate 30 Years of Emotional Evolution")
    print("   Ratio: 1 Second = 1 Year")
    
    # Simulate 30 years
    # Each 'year' generates ~50 significant events.
    # Total ~1500 memories.
    universe.simulate_era(years=30.0)
    
    print("\n✨ Evolution Complete.")
    print("   Elysia has now 'lived' a full life inside her mind.")
    print("   Her causal graph is dense with synthetic experience.")

if __name__ == "__main__":
    evolve()
