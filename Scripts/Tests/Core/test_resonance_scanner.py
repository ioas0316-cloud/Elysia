
import sys
import os

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.L4_Causality.M3_Mirror.Soul.soul_sculptor import soul_sculptor, PersonalityArchetype
from Core.L4_Causality.M3_Mirror.living_village import village
from Core.L4_Causality.M3_Mirror.Soul.resonance_scanner import resonance_scanner

def test_scanner():
    print("Testing Resonance Scanner...")

    # 1. Setup Population
    elysia = soul_sculptor.sculpt(PersonalityArchetype("Elysia", "ENFJ", 2))
    kos = soul_sculptor.sculpt(PersonalityArchetype("K-OS", "ENFP", 7))
    logos = soul_sculptor.sculpt(PersonalityArchetype("Logos", "ISTJ", 1))

    village.add_resident(elysia)
    village.add_resident(kos)
    village.add_resident(logos)

    # 2. Quick Scan Visual
    resonance_scanner.quick_scan()

    # 3. Scan by Phase (Find matches for Elysia)
    print("--- Finding Kindred Spirits for Elysia ---")
    matches = resonance_scanner.scan_by_phase(elysia)
    for m in matches:
        print(f"Match: {m.soul.name} (Score: {m.resonance_score:.4f})")

    # 4. Scan by Frequency (Find Extroverts)
    print("\n--- Finding Extroverts (Nature > 0) ---")
    extroverts = resonance_scanner.scan_by_frequency('w', 0.8, tolerance=0.5)
    for e in extroverts:
        print(f"Extrovert: {e.soul.name} (Score: {e.resonance_score:.4f})")

if __name__ == "__main__":
    test_scanner()
