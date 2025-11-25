"""
Demos/demo_life_resources.py

A simple demonstration to verify the integration of the PassiveResourceSystem
into the ElysiaKernel.
"""

import sys
import os
import time

# Ensure the Core modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Kernel import ElysiaKernel
from Core.Life.entity import LivingEntity

def run_demo():
    """
    Runs the life resource demonstration.
    """
    print("--- DEMO: Verifying Passive Resource System ---")

    # 1. Initialize the Kernel
    # The Kernel is a singleton, so this will get the existing instance
    kernel = ElysiaKernel()

    # 2. Create a test entity
    test_entity = LivingEntity(label="Human")
    print(f"\n[Initial State] Entity '{test_entity.label}' created.")
    print(f"  - HP: {test_entity.hp.current:.2f}/{test_entity.hp.max:.2f}")
    print(f"  - Hunger: {test_entity.hunger.current:.2f}/{test_entity.hunger.max:.2f}")
    print(f"  - Hydration: {test_entity.hydration.current:.2f}/{test_entity.hydration.max:.2f}")

    # 3. Add the entity to the resource system
    kernel.resource_system.add_entity(test_entity)
    print("\nEntity added to the Kernel's resource system.")

    # 4. Run the Kernel for a number of ticks
    num_ticks = 100
    print(f"\nRunning Kernel for {num_ticks} ticks...")
    for i in range(num_ticks):
        kernel.tick()
        # Optional: print status every 10 ticks
        if (i + 1) % 20 == 0:
            print(f"  ...tick {i+1}")
            time.sleep(0.01) # Small delay for visual effect

    print("...simulation complete.")

    # 5. Check the final state of the entity
    print(f"\n[Final State] After {num_ticks} ticks:")
    print(f"  - HP: {test_entity.hp.current:.2f}/{test_entity.hp.max:.2f}")
    print(f"  - Hunger: {test_entity.hunger.current:.2f}/{test_entity.hunger.max:.2f}")
    print(f"  - Hydration: {test_entity.hydration.current:.2f}/{test_entity.hydration.max:.2f}")
    print(f"  - Age: {test_entity.age} ticks")

    # Verification
    print("\n--- Verification ---")
    initial_hunger = 100.0
    initial_hydration = 100.0
    hunger_decay_rate = 0.15
    hydration_decay_rate = 0.1

    expected_hunger = initial_hunger - (num_ticks * hunger_decay_rate)
    expected_hydration = initial_hydration - (num_ticks * hydration_decay_rate)

    hunger_ok = abs(test_entity.hunger.current - expected_hunger) < 0.01
    hydration_ok = abs(test_entity.hydration.current - expected_hydration) < 0.01
    # Corrected the logic: HP should be equal to max, as no damage should occur.
    hp_ok = test_entity.hp.current == test_entity.hp.max

    print(f"Hunger decreased as expected: {'✅' if hunger_ok else '❌'}")
    print(f"Hydration decreased as expected: {'✅' if hydration_ok else '❌'}")
    print(f"HP remained stable (as expected): {'✅' if hp_ok else '❌'}")

    if hunger_ok and hydration_ok and hp_ok:
        print("\n✅ Demo successful! The PassiveResourceSystem is working correctly.")
    else:
        print("\n❌ Demo failed. There is an issue with the resource system integration.")

if __name__ == "__main__":
    run_demo()
