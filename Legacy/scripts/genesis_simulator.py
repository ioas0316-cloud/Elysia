
import logging
import os
import sys
import numpy as np
from unittest.mock import MagicMock

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics

def main():
    """
    A sandbox for verifying the newly implemented threat law.
    This scenario tests if the 'h_imprint' (historical imprint) of a death
    contributes to the 'threat_field', causing other cells to avoid the location.
    """
    # --- 0. Logging Setup ---
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    file_handler = logging.FileHandler("threat_law_test.log", mode='w')
    file_handler.setFormatter(log_formatter)

    # --- 1. Initialization ---
    root_logger.info("Initializing Simulator for the Threat Law...")
    world = World(
        primordial_dna={'instinct': 'survive'},
        wave_mechanics=MagicMock(spec=WaveMechanics),
        logger=logging.getLogger('WorldLogger')
    )
    world.logger.addHandler(file_handler)
    world.logger.setLevel(logging.INFO)
    world.logger.propagate = False

    # --- 2. World Creation: A simple scene with two humans ---
    root_logger.info("Creating a scene with a 'survivor' and a 'victim'.")

    world.add_cell('survivor', properties={'label': 'human', 'element_type': 'animal', 'diet': 'omnivore', 'position': {'x': 50, 'y': 50, 'z': 0}})
    world.add_cell('victim', properties={'label': 'human', 'element_type': 'animal', 'diet': 'omnivore', 'position': {'x': 52, 'y': 50, 'z': 0}})

    survivor_idx = world.id_to_idx['survivor']
    victim_idx = world.id_to_idx['victim']

    root_logger.info("Initial world state:")
    world.print_world_summary()

    # --- 3. Simulation Loop ---
    num_steps = 20
    death_step = 5
    death_pos = np.copy(world.positions[victim_idx])
    death_pos_grid = (int(death_pos[1]), int(death_pos[0])) # (y, x)

    root_logger.info(f"Running simulation for {num_steps} steps. Victim will die at step {death_step} at position {death_pos[:2]}.")

    initial_distance = np.linalg.norm(world.positions[survivor_idx] - death_pos)

    for i in range(num_steps):
        print(f"\n--- Running Step {i+1} ---")

        if i == death_step:
            root_logger.info(f"--- KILLING VICTIM AT STEP {i+1} ---")
            world.hp[victim_idx] = 0
            # Manually trigger cleanup to ensure death is processed and h_imprint is created
            world._apply_physics_and_cleanup([])
            root_logger.info("Victim has been killed. An 'h_imprint' should now exist at their death location.")

        world.run_simulation_step()
        world.print_world_summary()

        # --- Verification ---
        survivor_pos = world.positions[survivor_idx]
        current_distance = np.linalg.norm(survivor_pos - death_pos)

        if i > death_step:
            threat_at_death_spot = world.threat_field[death_pos_grid]
            h_imprint_at_death_spot = world.h_imprint[death_pos_grid]

            root_logger.info(f"VERIFICATION: Survivor Position: {survivor_pos[:2]}, Distance from death spot: {current_distance:.2f}")
            root_logger.info(f"VERIFICATION: h_imprint at death spot: {h_imprint_at_death_spot:.4f}, Threat at death spot: {threat_at_death_spot:.4f}")

            if threat_at_death_spot > 0:
                 root_logger.info(f"VERIFICATION SUCCESS: h_imprint is generating a threat field.")
            if current_distance > initial_distance + 0.1: # Check if survivor is moving away
                 root_logger.info(f"VERIFICATION SUCCESS: Survivor is moving away from the death spot.")

        initial_distance = current_distance


    root_logger.info("Threat Law simulation complete. Check 'threat_law_test.log' for detailed behavior.")

if __name__ == "__main__":
    main()
