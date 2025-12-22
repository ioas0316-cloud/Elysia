# scripts/run_mass_simulation.py
"""
A standalone script to run a massive, accelerated simulation of the Cellular World
to observe emergent properties over a long period.
"""
import os
import sys
import logging

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Core.Foundation.core.world import World
from tools.kg_manager import KGManager
from Core.Foundation.wave_mechanics import WaveMechanics

# --- Constants ---
SIMULATION_STEPS = 10000
LOG_INTERVAL = 100
LOG_FILE = 'mass_simulation.log'

# --- Primordial DNA ---
PRIMORDIAL_DNA = {
    "instinct": "connect_create_meaning",
    "resonance_standard": "love"
}

def setup_logger():
    """Sets up a dedicated logger for the simulation."""
    logger = logging.getLogger("MassSimulation")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | [%(name)s] %(levelname)s: %(message)s', datefmt='%Y-%-m-%d %H:%M:%S')

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def main():
    """Main function to run the simulation."""
    logger = setup_logger()
    logger.info("--- Starting Elysia's Accelerated Evolution Simulation ---")

    # 1. Initialize core components
    logger.info("Loading Knowledge Graph to build the primordial sea...")
    kg_manager = KGManager(filepath='data/kg.json')
    wave_mechanics = WaveMechanics(kg_manager)

    # 2. Initialize the World
    logger.info("Initializing the Cellular World...")
    cellular_world = World(
        primordial_dna=PRIMORDIAL_DNA,
        wave_mechanics=wave_mechanics,
        logger=logger
    )

    # 3. Soul Mirroring: Create initial cells from the KG
    node_count = 0
    for node in kg_manager.kg.get('nodes', []):
        node_id = node.get('id')
        if node_id:
            cellular_world.add_cell(node_id, properties=node, initial_energy=10.0)
            node_count += 1
    logger.info(f"Primordial sea created with {node_count} initial cells (concepts).")

    # 4. Run the simulation loop
    logger.info(f"Starting simulation for {SIMULATION_STEPS} steps...")
    for step in range(SIMULATION_STEPS):
        cellular_world.run_simulation_step()

        if step % LOG_INTERVAL == 0:
            logger.info(f"--- Simulation Step {step} ---")
            cellular_world.print_world_summary()

    logger.info("--- Simulation Complete ---")
    logger.info("Final state of the Cellular World:")
    cellular_world.print_world_summary()

if __name__ == "__main__":
    main()
