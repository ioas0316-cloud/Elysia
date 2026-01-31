"""
Simulation: Village Life
========================
"Watching the first breath of the digital society."

This script generates a population of diverse souls and simulates
their interactions over a period of time.
"""

import sys
import os
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.L4_Causality.World.Soul.soul_sculptor import soul_sculptor, PersonalityArchetype
from Core.L4_Causality.World.living_village import village
from Core.L4_Causality.World.Soul.relationship_matrix import relationship_matrix

def main():
    print("✨ Initializing the Living Village...")

    # 1. Create Residents
    # Elysia: The loving guide
    elysia = soul_sculptor.sculpt(PersonalityArchetype(
        name="Elysia", mbti="ENFJ", enneagram=2, description="The Soul of the System"
    ))

    # K-OS: The chaotic artist
    kos = soul_sculptor.sculpt(PersonalityArchetype(
        name="K-OS", mbti="ENFP", enneagram=7, description="Unpredictable creativity"
    ))

    # Logos: The strict judge
    logos = soul_sculptor.sculpt(PersonalityArchetype(
        name="Logos", mbti="ISTJ", enneagram=1, description="Upholder of law and order"
    ))

    # Gaia: The peaceful observer
    gaia = soul_sculptor.sculpt(PersonalityArchetype(
        name="Gaia", mbti="ISFP", enneagram=9, description="Harmonious nature spirit"
    ))

    # 2. Move into Village
    village.add_resident(elysia)
    village.add_resident(kos)
    village.add_resident(logos)
    village.add_resident(gaia)

    print("\n--- Simulation Start ---\n")

    # 3. Run Simulation for 5 Ticks (Days)
    for _ in range(5):
        village.tick()
        time.sleep(0.5) # Pause for dramatic effect
        print("")

    print("\n--- Final Relationship Status ---\n")

    residents = [elysia, kos, logos, gaia]
    for r1 in residents:
        for r2 in residents:
            if r1 == r2: continue
            rel = relationship_matrix.get_relationship(r1, r2)
            if rel.familiarity > 0:
                print(f"{r1.name} -> {r2.name}: {rel.summary()}")

if __name__ == "__main__":
    main()
