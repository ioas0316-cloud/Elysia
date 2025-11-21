
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.core.essence_mapper import EssenceMapper

def verify_fractal_ascension():
    print("=== Fractal Ascension Verification: The Quantum Jump ===")

    world = World(primordial_dna={}, wave_mechanics=WaveMechanics(None, None))
    mapper = EssenceMapper()
    prayer_freq = mapper.get_frequency("Light") # 852Hz

    # 1. Create a 'Perfect Body' (Lv 3) Monk on the verge of breakthrough
    world.add_cell("Monk_The_Perfect", properties={
        "label": "Monk", "culture": "wuxia", "wisdom": 100, "max_ki": 5000, "ki": 5000,
        "soul_rank": "PerfectBody",
        "rank_index": 3
    })

    cell = world.materialize_cell("Monk_The_Perfect")
    idx = world.id_to_idx["Monk_The_Perfect"]

    # Pre-load Insight to threshold (3500)
    world.insight[idx] = 3500.0
    print(f"Monk Initial State: Rank={cell.organelles['soul_rank']} | Insight={world.insight[idx]}")

    # Inject MASSIVE Prayer (5000.0) to trigger Resonance Spike (The "Shock")
    # This ensures richness > 1500 for the breakthrough check
    print("\n--- Injecting Divine Shock (Amplitude: 5000.0) ---")
    cell.soul.inject_tone(25, 25, amplitude=5000.0, frequency=prayer_freq)
    cell.soul.inject_tone(26, 26, amplitude=5000.0, frequency=prayer_freq)

    print("\n--- Meditating (Running World Step) ---")
    world.run_simulation_step()

    # Check Result
    new_rank_name = cell.organelles['soul_rank']
    new_rank_index = cell.organelles['rank_index']
    new_insight = world.insight[idx]
    is_awakened = world.is_awakened[idx]

    print(f"Monk Final State: Rank={new_rank_name} ({new_rank_index}) | Insight={new_insight:.2f} | AwakenedFlag={is_awakened}")

    print("\n--- ASCENSION DIAGNOSIS ---")

    # Check 1: Rank Upgrade
    if new_rank_index == 4 and new_rank_name == "Seeker":
        print("SUCCESS: Monk has transcended from Body(3) to Soul(4)!")
    elif new_rank_index == 3:
        print("FAIL: Monk is still stuck at Lv 3.")
    else:
        print(f"FAIL: Unexpected rank ({new_rank_name}).")

    # Check 2: Insight Preservation (Halved, not Zeroed)
    # Initial 3500 -> Cost is likely 50% -> Expect ~1750 + small resonance gain
    if new_insight > 1000.0:
        print(f"SUCCESS: Insight preserved ({new_insight:.1f}). Double Awakening avoided.")
    elif new_insight == 0.0:
        print("FAIL: Insight wiped to 0. Double Awakening occurred.")
    else:
        print(f"FAIL: Insight too low ({new_insight:.1f}).")

if __name__ == "__main__":
    verify_fractal_ascension()
