
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

    # Use a temporary KG file for this simulation to avoid side-effects
    temp_kg_path = "temp_genesis_kg.json"
    kg_manager = KGManager(filepath=temp_kg_path)
    kg_manager.kg = {"nodes": [], "edges": []} # Start with a clean slate

    # Populate the KG with the essential nodes for our simulation
    kg_manager.add_node('sun', properties={'activation_energy': 2.0, 'element_type': 'nature'})
    kg_manager.add_node('love', properties={'activation_energy': 1.0, 'element_type': 'emotion'})
    kg_manager.add_node('plant', properties={'element_type': 'life'})
    kg_manager.add_node('water', properties={'element_type': 'nature'})
    kg_manager.add_node('earth', properties={'element_type': 'nature'})
    kg_manager.add_node('wolf', properties={'element_type': 'animal'})

    wave_mechanics = WaveMechanics(kg_manager)
    world = World(
        primordial_dna={'instinct': 'survive_and_grow'},
        wave_mechanics=wave_mechanics,
        logger=logger
    )

    # --- 2. World Creation ---
    logger.info("Creating a micro-world for the experiment...")
    world.add_cell('plant_A', initial_energy=10.0)
    world.add_cell('water_A', initial_energy=10.0)
    world.add_cell('earth_A', initial_energy=10.0)
    world.add_cell('wolf_A', initial_energy=20.0) # Start with more energy

    world.add_connection('plant_A', 'water_A', strength=0.5)
    world.add_connection('plant_A', 'earth_A', strength=0.5)
    world.add_connection('wolf_A', 'plant_A', strength=0.5) # The wolf is near the plant

    # Manually set element types for the simulation
    world.element_types[world.id_to_idx['plant_A']] = 'life'
    world.element_types[world.id_to_idx['water_A']] = 'nature'
    world.element_types[world.id_to_idx['earth_A']] = 'nature'
    world.element_types[world.id_to_idx['wolf_A']] = 'animal'

    logger.info("Initial world state:")
    world.print_world_summary()

    # --- 3. Simulation Loop ---
    num_steps = 5
    logger.info(f"Running simulation for {num_steps} steps...")
    for i in range(num_steps):
        print(f"\n--- Running Step {i+1} ---")
        world.run_simulation_step()
        world.print_world_summary()

    logger.info("Genesis simulation complete.")

if __name__ == "__main__":
    main()
