
import logging
import os
import sys
import random
from unittest.mock import MagicMock

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Core.FoundationLayer.Foundation.core.world import World
from Core.FoundationLayer.Foundation.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager

def create_dawn_of_civilization_scene(world: World):
    """Creates a scenario with two distinct human cultures on different continents."""
    # --- Eastern Continent: Wuxia Culture ---
    world.add_cell("wuxia_master", properties={
        'element_type': 'animal', 'diet': 'omnivore', 'label': 'human', 'gender': 'male',
        'continent': 'East', 'culture': 'wuxia'}, initial_energy=50.0)
    world.add_cell("wuxia_student", properties={
        'element_type': 'animal', 'diet': 'omnivore', 'label': 'human', 'gender': 'female',
        'continent': 'East', 'culture': 'wuxia'}, initial_energy=20.0)

    # --- Western Continent: Knight Culture ---
    world.add_cell("knight_captain", properties={
        'element_type': 'animal', 'diet': 'omnivore', 'label': 'human', 'gender': 'male',
        'continent': 'West', 'culture': 'knight'}, initial_energy=50.0)
    world.add_cell("knight_squire", properties={
        'element_type': 'animal', 'diet': 'omnivore', 'label': 'human', 'gender': 'female',
        'continent': 'West', 'culture': 'knight'}, initial_energy=20.0)

    # Environment
    world.add_cell("eastern_plains", properties={'element_type': 'earth', 'continent': 'East'}, initial_energy=100.0)
    world.add_cell("western_castle", properties={'element_type': 'item', 'continent': 'West'}, initial_energy=100.0)

    # Establish connections
    world.add_connection("wuxia_master", "wuxia_student", strength=0.9)
    world.add_connection("knight_captain", "knight_squire", strength=0.9) # Strengthened bond

    # Manually set positions for clarity
    world.positions[world.id_to_idx['wuxia_master']] = [-50, 0, 0]
    world.positions[world.id_to_idx['wuxia_student']] = [-49, 0, 0]
    world.positions[world.id_to_idx['knight_captain']] = [50, 0, 0]
    world.positions[world.id_to_idx['knight_squire']] = [49, 0, 0]


def main():
    """Sets up and runs the customized simulation for the dawn of civilizations."""
    # --- 0. Detailed Logging Setup ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Initializing Civilization Simulator for the Dawn of a New Age...")

    # --- 1. Initialization ---
    kg_manager = KGManager()
    wave_mechanics = WaveMechanics(kg_manager)
    world = World(primordial_dna={}, wave_mechanics=wave_mechanics, logger=logger)

    # --- 2. World Creation ---
    logger.info("Creating a scene for the dawn of civilization...")
    create_dawn_of_civilization_scene(world)

    logger.info("Initial world state:")
    world.print_world_summary()

    # --- 3. Simulation Loop ---
    num_steps = 10 # Run for a few steps to observe initial state
    logger.info(f"Running simulation for {num_steps} steps...")
    for i in range(num_steps):
        print(f"\n--- Running Step {i+1} ---")
        world.run_simulation_step()
        world.print_world_summary()

    logger.info("Civilization simulation complete.")

if __name__ == "__main__":
    main()
