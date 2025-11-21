
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Sophia.core.essence_mapper import EssenceMapper

def verify_ensoulment_balance():
    print("=== Ensoulment Balance Verification: The Law of Capacity ===")

    world = World(primordial_dna={}, wave_mechanics=WaveMechanics(None, None))
    mapper = EssenceMapper()
    prayer_freq = mapper.get_frequency("Light") # 852Hz

    # 1. Create 'Sleeping Monk' (Rank 0)
    # Should filter out most noise.
    world.add_cell("Sleeping_Monk", properties={
        "label": "Monk", "culture": "wuxia", "wisdom": 80, "max_ki": 100, "ki": 10,
        "vessel_rank": 0
    })

    # 2. Create 'Awakened Monk' (Rank 1)
    # Should gain resonance logarithmically.
    world.add_cell("Awakened_Monk", properties={
        "label": "Monk", "culture": "wuxia", "wisdom": 80, "max_ki": 100, "ki": 10,
        "vessel_rank": 1
    })

    # 3. Create 'Avatar Monk' (Incarnate)
    # Should gain massive, linear power.
    world.add_cell("Avatar_Monk", properties={
        "label": "Monk", "culture": "wuxia", "wisdom": 80, "max_ki": 1000, "ki": 10,
        "is_avatar": True
    })

    # Materialize and Inject Prayer
    for cell_id in ["Sleeping_Monk", "Awakened_Monk", "Avatar_Monk"]:
        cell = world.materialize_cell(cell_id)
        # Manual Ki drain for test accuracy
        world.ki[world.id_to_idx[cell_id]] = 10.0
        # Inject strong prayer
        cell.soul.inject_tone(25, 30, amplitude=1.0, frequency=prayer_freq)

    print("\n--- Meditating (Running World Steps) ---")
    # Run enough steps to build up significant richness
    for i in range(1, 11):
        world.run_simulation_step()

        if i == 10:
            print(f"\n[Step {i} Report]")
            for cell_id in ["Sleeping_Monk", "Awakened_Monk", "Avatar_Monk"]:
                idx = world.id_to_idx[cell_id]
                ki = world.ki[idx]
                insight = world.insight[idx]
                richness = world.materialized_cells[cell_id].soul.grid[:, :, 2].sum()
                hp = world.hp[idx]

                print(f"  > {cell_id}: Richness={richness:.0f} | Ki={ki:.1f} | Insight={insight:.2f} | HP={hp:.1f}")

    # Verify Logic
    s_idx = world.id_to_idx["Sleeping_Monk"]
    w_idx = world.id_to_idx["Awakened_Monk"]
    a_idx = world.id_to_idx["Avatar_Monk"]

    s_gain = world.ki[s_idx] - 10.0
    w_gain = world.ki[w_idx] - 10.0
    a_gain = world.ki[a_idx] - 10.0

    print("\n--- BALANCE DIAGNOSIS ---")

    # Avatar should vastly overpower Awakened
    if a_gain > w_gain * 10:
        print("SUCCESS: Avatar power is orders of magnitude higher (Linear vs Log).")
    else:
        print(f"FAIL: Avatar power ratio insufficient ({a_gain:.1f} vs {w_gain:.1f}).")

    # Awakened should be better than Sleeping
    if w_gain > s_gain:
        print("SUCCESS: Awakened vessel channels more than Sleeping vessel.")
    else:
        print("FAIL: Awakened not superior to Sleeping.")

    # Sleeping should be minimal
    if s_gain < 100:
        print("SUCCESS: Sleeping vessel filters out cosmic noise effectively.")
    else:
        print(f"FAIL: Sleeping vessel leaking too much power ({s_gain:.1f}).")

if __name__ == "__main__":
    verify_ensoulment_balance()
