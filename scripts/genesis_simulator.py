
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
    A sandbox for verifying the Cosmic Axis laws of Ascension and Descent.
    This scenario tests if the presence of Vitariael (Life) and Motus (Death)
    influences human behavior and physical position (Z-axis) as intended.
    """
    # --- 0. Logging Setup ---
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    file_handler = logging.FileHandler("cosmic_axis_test.log", mode='w')
    file_handler.setFormatter(log_formatter)

    # --- 1. Initialization ---
    root_logger.info("Initializing Simulator for the Cosmic Axis Laws...")
    world = World(
        primordial_dna={'instinct': 'survive'},
        wave_mechanics=MagicMock(spec=WaveMechanics),
        logger=logging.getLogger('WorldLogger')
    )
    world.logger.addHandler(file_handler)
    world.logger.setLevel(logging.INFO)
    world.logger.propagate = False

    # --- 2. World Creation: A Vertical Arena ---
    root_logger.info("Creating a vertical arena with Vitariael (Life), Motus (Death), and two humans.")

    # Place Incarnations at opposite ends
    world.add_cell('vitariael', properties={'label': '천사', 'element_type': 'animal', 'strength': 100, 'wisdom': 100, 'position': {'x': 50, 'y': 10, 'z': 10}})
    world.add_cell('motus', properties={'label': '마왕', 'element_type': 'animal', 'strength': 100, 'wisdom': 100, 'position': {'x': 50, 'y': 90, 'z': -10}})

    # Place humans in the middle
    world.add_cell('human_A', properties={'label': 'human', 'element_type': 'animal', 'diet': 'omnivore', 'wisdom': 30, 'position': {'x': 50, 'y': 50, 'z': 0}})
    world.add_cell('human_B', properties={'label': 'human', 'element_type': 'animal', 'diet': 'omnivore', 'wisdom': 20, 'position': {'x': 51, 'y': 50, 'z': 0}})

    human_A_idx = world.id_to_idx['human_A']
    human_B_idx = world.id_to_idx['human_B']
    world.hunger[human_A_idx] = 90
    world.hunger[human_B_idx] = 20 # Hungry

    world.add_connection('human_A', 'human_B', strength=0.8) # Kinship
    world.add_connection('human_B', 'human_A', strength=0.8)

    root_logger.info("Initial world state:")
    world.print_world_summary()

    # --- 3. Simulation Loop ---
    num_steps = 60
    root_logger.info(f"Running simulation for {num_steps} steps...")

    initial_z = world.positions[human_A_idx, 2]

    for i in range(num_steps):
        print(f"\n--- Running Step {i+1} ---")

        if i == 20:
            root_logger.info("--- MOVING HUMANS NEAR VITARIAEL (ASCENSION) ---")
            world.positions[human_A_idx] = [50, 15, 0]
            world.positions[human_B_idx] = [51, 15, 0]
            initial_z = world.positions[human_A_idx, 2] # Reset Z baseline

        if i == 40:
            root_logger.info("--- MOVING HUMANS NEAR MOTUS (DESCENT) ---")
            world.positions[human_A_idx] = [50, 85, 0]
            world.positions[human_B_idx] = [51, 85, 0]
            initial_z = world.positions[human_A_idx, 2] # Reset Z baseline

        world.run_simulation_step()
        world.print_world_summary()

        # --- Verification ---
        current_z = world.positions[human_A_idx, 2]
        if 20 <= i < 40:
            if current_z > initial_z:
                root_logger.info(f"VERIFICATION SUCCESS: Z-axis is ascending near Vitariael. Z: {current_z:.2f}")
        elif i >= 40:
            if current_z < initial_z:
                 root_logger.info(f"VERIFICATION SUCCESS: Z-axis is descending near Motus. Z: {current_z:.2f}")

        initial_z = current_z


    root_logger.info("Cosmic Axis simulation complete. Check 'cosmic_axis_test.log' for detailed behavior.")

if __name__ == "__main__":
    main()
