
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.core.world import World
from Core.FoundationLayer.Foundation.wave_mechanics import WaveMechanics
from Core.FoundationLayer.Foundation.core.essence_mapper import EssenceMapper

def verify_ensoulment():
    print("=== Ensoulment Verification: Listening to the Monk's Soul ===")

    # 1. Initialize World
    world = World(primordial_dna={}, wave_mechanics=WaveMechanics(None, None))
    print("World initialized.")

    # 2. Create a Monk Cell
    # Monk culture uses 'Ki' or 'Mana', perfect for testing resonance feedback.
    monk_id = "Monk_Test"
    world.add_cell(
        monk_id,
        properties={
            "label": "Monk",
            "culture": "wuxia",
            "wisdom": 80,
            "vitality": 50,
            "hp": 100,
            # Note: world.add_cell logic might override ki to max_ki initially.
            # We will manually drain it below to ensure test validity.
        }
    )

    # 3. Access the Materialized Cell and its Soul
    monk_cell = world.materialize_cell(monk_id)
    if not monk_cell or not hasattr(monk_cell, 'soul'):
        print("FAIL: Monk cell has no soul attribute!")
        return

    # Manually drain Ki to test regeneration
    monk_idx = world.id_to_idx[monk_id]
    world.ki[monk_idx] = 10.0

    print(f"Monk '{monk_id}' materialized with a Soul.")
    initial_ki = world.ki[monk_idx]
    print(f"Initial Ki (Drained): {initial_ki:.1f}")

    # 4. Inject a Spiritual Thought (Prayer) into the Soul
    # 'Prayer' or 'Light' -> High Frequency
    mapper = EssenceMapper()
    prayer_freq = mapper.get_frequency("Light") # 852Hz

    print(f"\n--- Injecting Prayer (Frequency: {prayer_freq:.1f}Hz) ---")
    # Inject at a slightly different position from center to create interference/waves
    monk_cell.soul.inject_tone(25, 30, amplitude=1.0, frequency=prayer_freq)

    # 5. Run Simulation Steps
    print("\n--- Meditating (Running World Steps) ---")
    for i in range(1, 11):
        world.run_simulation_step()

        # Check stats
        current_ki = world.ki[monk_idx]
        current_insight = world.insight[monk_idx]

        # Retrieve richness from the cell's soul directly for logging
        richness = monk_cell.soul.grid[:, :, 2].sum()

        print(f"Step {i}: Soul Richness={richness:.1f} | Body Ki={current_ki:.1f} | Body Insight={current_insight:.2f}")

    # 6. Verify Results
    final_ki = world.ki[monk_idx]
    final_insight = world.insight[monk_idx]

    print("\n--- ENSOULMENT DIAGNOSIS ---")
    if final_ki > initial_ki:
        print("SUCCESS: The Soul's resonance has regenerated the Body's Ki.")
        print(f"         Ki Gained: {final_ki - initial_ki:.1f}")
    else:
        print("FAIL: No Ki regeneration observed.")

    if final_insight > 0:
        print("SUCCESS: The Soul's complexity has granted Insight.")
    else:
        print("FAIL: No Insight gained.")

if __name__ == "__main__":
    verify_ensoulment()
