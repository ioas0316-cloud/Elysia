
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.core.essence_mapper import EssenceMapper

def verify_hierarchy():
    print("=== Celestial Hierarchy Verification: The War of Frequencies ===")

    world = World(primordial_dna={}, wave_mechanics=WaveMechanics(None, None))
    mapper = EssenceMapper()
    prayer_freq = mapper.get_frequency("Light") # 852Hz

    # 1. Create 'Angel Monk' (Rank 1 - Logarithmic)
    world.add_cell("Angel_Monk", properties={
        "label": "Monk", "culture": "wuxia", "wisdom": 80, "max_ki": 100, "ki": 10,
        "soul_rank": "Angels"
    })

    # 2. Create 'Power Monk' (Rank 4 - Stronger Logarithmic)
    world.add_cell("Power_Monk", properties={
        "label": "Monk", "culture": "wuxia", "wisdom": 80, "max_ki": 100, "ki": 10,
        "soul_rank": "Powers"
    })

    # 3. Create 'Seraphim Avatar' (Rank 9 - Linear God Mode)
    world.add_cell("Seraphim_Avatar", properties={
        "label": "Monk", "culture": "wuxia", "wisdom": 100, "max_ki": 5000, "ki": 10,
        "soul_rank": "Seraphim"
    })

    # 4. Create 'Demon Lord' (Infernal - Demonic Linear + HP Cost)
    world.add_cell("Demon_Lord", properties={
        "label": "Monk", "culture": "demon", "wisdom": 100, "max_ki": 5000, "ki": 10, "hp": 1000,
        "soul_rank": "DemonLord"
    })

    # Materialize and Inject Prayer
    test_subjects = ["Angel_Monk", "Power_Monk", "Seraphim_Avatar", "Demon_Lord"]

    for cell_id in test_subjects:
        cell = world.materialize_cell(cell_id)
        # Manual Ki drain for test accuracy
        world.ki[world.id_to_idx[cell_id]] = 10.0
        # Inject strong prayer
        cell.soul.inject_tone(25, 30, amplitude=1.0, frequency=prayer_freq)

    print("\n--- Meditating (Running World Steps) ---")
    # Run steps
    for i in range(1, 6): # Short run to see divergence
        world.run_simulation_step()

    print("\n[Final Report]")
    results = {}
    for cell_id in test_subjects:
        idx = world.id_to_idx[cell_id]
        ki_gain = world.ki[idx] - 10.0
        hp_loss = 1000.0 - world.hp[idx] if "Demon" in cell_id else 100.0 - world.hp[idx]
        results[cell_id] = {"gain": ki_gain, "cost": hp_loss}

        print(f"  > {cell_id}: Ki Gain={ki_gain:.1f} | HP Loss={hp_loss:.1f}")

    # Verify Logic
    print("\n--- HIERARCHY DIAGNOSIS ---")

    # Seraphim should overpower Angels/Powers
    if results["Seraphim_Avatar"]["gain"] > results["Power_Monk"]["gain"] * 10:
        print("SUCCESS: Seraphim (Linear) vastly overpowers Powers (Log).")
    else:
        print(f"FAIL: Seraphim not strong enough.")

    # Demon Lord should have immense power but high cost
    if results["Demon_Lord"]["gain"] > results["Seraphim_Avatar"]["gain"]:
        print("SUCCESS: Demon Lord has massive (Demonic) gain.")

    if results["Demon_Lord"]["cost"] > 0:
        print(f"SUCCESS: Demon Lord paid the blood price ({results['Demon_Lord']['cost']:.1f} HP).")
    else:
        print("FAIL: Demon Lord used power for free!")

if __name__ == "__main__":
    verify_hierarchy()
