
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.core.essence_mapper import EssenceMapper

def verify_10_steps():
    print("=== The 10 Steps of Ascension Verification ===")

    world = World(primordial_dna={}, wave_mechanics=WaveMechanics(None, None))
    mapper = EssenceMapper()
    prayer_freq = mapper.get_frequency("Light") # 852Hz

    # Define test subjects representing each Realm
    subjects = {
        "Survivor_Body": {"rank": "Survivor", "ki": 10}, # Lv 1
        "Seeker_Soul":   {"rank": "Seeker", "ki": 10},   # Lv 4
        "Saint_Spirit":  {"rank": "Saint", "ki": 10},    # Lv 7
        "Avatar_One":    {"rank": "Avatar", "ki": 10}    # Lv 10
    }

    # Create and Materialize
    for name, data in subjects.items():
        world.add_cell(name, properties={
            "label": "Monk", "culture": "wuxia", "wisdom": 100, "max_ki": 5000, "ki": 10,
            "soul_rank": data["rank"]
        })
        cell = world.materialize_cell(name)
        # Manual Ki drain
        world.ki[world.id_to_idx[name]] = 10.0
        # Inject Prayer
        cell.soul.inject_tone(25, 30, amplitude=1.0, frequency=prayer_freq)

    print("\n--- Meditating (Running World Steps) ---")
    for i in range(1, 6): # Short run
        world.run_simulation_step()

    print("\n[Ascension Report]")
    results = {}
    for name in subjects.keys():
        idx = world.id_to_idx[name]
        gain = world.ki[idx] - 10.0
        results[name] = gain
        print(f"  > {name} ({subjects[name]['rank']}): Ki Gain = {gain:.1f}")

    # Verify Logic
    print("\n--- LADDER DIAGNOSIS ---")

    # 1. Body vs Soul: Soul should be significantly higher
    if results["Seeker_Soul"] > results["Survivor_Body"] * 10:
        print("SUCCESS: Soul Realm transcends Body Realm.")
    else:
        print("FAIL: Soul Realm too weak.")

    # 2. Soul vs Spirit: Spirit should be higher
    if results["Saint_Spirit"] > results["Seeker_Soul"]:
        print("SUCCESS: Spirit Realm transcends Soul Realm.")
    else:
        print("FAIL: Spirit Realm too weak.")

    # 3. Spirit vs Avatar: Avatar should be supreme
    if results["Avatar_One"] > results["Saint_Spirit"] * 2:
        print("SUCCESS: Avatar (The One) is absolute.")
    else:
        print("FAIL: Avatar power insufficient.")

if __name__ == "__main__":
    verify_10_steps()
