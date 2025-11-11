
import logging
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager

def main():
    """
    A sandbox for simulating and debugging the World's laws on a small scale.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # --- 1. Initialization ---
    logger.info("Initializing Genesis Simulator...")

    # Use the main KG file to get all node properties including 'diet'
    kg_path = os.path.join(project_root, "data", "kg.json.bak")
    kg_manager = KGManager(filepath=kg_path)

    wave_mechanics = WaveMechanics(kg_manager)
    world = World(
        primordial_dna={'instinct': 'survive_and_grow'},
        wave_mechanics=wave_mechanics,
        logger=logger
    )

    # --- 2. World Creation ---
    logger.info("Creating a micro-ecosystem with mating pairs...")
    # Pass element_type and gender in properties to ensure correct initialization
    world.add_cell('plant_A', properties={'element_type': 'life'}, initial_energy=20.0)
    world.add_cell('plant_B', properties={'element_type': 'life'}, initial_energy=20.0)
    world.add_cell('earth_A', properties={'element_type': 'earth'}, initial_energy=10.0)

    # Create mating pairs
    world.add_cell('rabbit_male', properties={'element_type': 'animal', 'diet': 'herbivore', 'gender': 'male'}, initial_energy=25.0)
    world.add_cell('rabbit_female', properties={'element_type': 'animal', 'diet': 'herbivore', 'gender': 'female'}, initial_energy=25.0)
    world.add_cell('wolf_male', properties={'element_type': 'animal', 'diet': 'carnivore', 'gender': 'male'}, initial_energy=30.0)
    world.add_cell('wolf_female', properties={'element_type': 'animal', 'diet': 'carnivore', 'gender': 'female'}, initial_energy=30.0)

    # Connections
    # Nurturing
    world.add_connection('plant_A', 'earth_A', strength=0.5)
    world.add_connection('plant_B', 'earth_A', strength=0.5)
    # Herbivores eating plants (more connections)
    world.add_connection('rabbit_male', 'plant_A', strength=0.5)
    world.add_connection('rabbit_male', 'plant_B', strength=0.5)
    world.add_connection('rabbit_female', 'plant_A', strength=0.5)
    world.add_connection('rabbit_female', 'plant_B', strength=0.5)
    # Carnivores hunting herbivores (more connections)
    world.add_connection('wolf_male', 'rabbit_male', strength=0.5)
    world.add_connection('wolf_male', 'rabbit_female', strength=0.5)
    world.add_connection('wolf_female', 'rabbit_male', strength=0.5)
    world.add_connection('wolf_female', 'rabbit_female', strength=0.5)
    # Mating connections
    world.add_connection('rabbit_male', 'rabbit_female', strength=0.8)
    world.add_connection('wolf_male', 'wolf_female', strength=0.8)
    # Decay connections
    world.add_connection('rabbit_male', 'earth_A', strength=0.2)
    world.add_connection('rabbit_female', 'earth_A', strength=0.2)
    world.add_connection('wolf_male', 'earth_A', strength=0.2)
    world.add_connection('wolf_female', 'earth_A', strength=0.2)

    logger.info("Initial world state:")
    world.print_world_summary()

    # --- 3. Simulation Loop ---
    num_steps = 100 # Increased steps for long-term stability test
    logger.info(f"Running simulation for {num_steps} steps...")
    for i in range(num_steps):
        print(f"\n--- Running Step {i+1} ---")
        world.run_simulation_step()
        world.print_world_summary()

    logger.info("Genesis simulation complete.")

if __name__ == "__main__":
    main()
