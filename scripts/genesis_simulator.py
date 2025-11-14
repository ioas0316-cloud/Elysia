
import logging
import os
import sys
import random
from unittest.mock import MagicMock

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager

def main():
    """
    A sandbox for simulating and debugging the Dawn of Humanity,
    testing the emergence of insight, tool use, and protection.
    """
    # --- 0. Detailed Logging Setup ---
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Root logger for console output
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # File logger for detailed simulation log
    file_handler = logging.FileHandler("genesis_final.log", mode='w')
    file_handler.setFormatter(log_formatter)

    # --- 1. Initialization ---
    root_logger.info("Initializing Genesis Simulator for the Dawn of Humanity...")

    kg_path = os.path.join(project_root, "data", "kg.json.bak")
    kg_manager = KGManager(filepath=kg_path)

    # Mock WaveMechanics to observe the stimulus injections
    mock_wave_mechanics = MagicMock(spec=WaveMechanics)
    mock_wave_mechanics.kg_manager = kg_manager

    world = World(
        primordial_dna={'instinct': 'survive_and_grow'},
        wave_mechanics=mock_wave_mechanics,
        logger=logging.getLogger('Project_Sophia.core.world')
    )
    # Ensure the world's logger writes to our file for detailed debugging
    world.logger.addHandler(file_handler)
    world.logger.setLevel(logging.INFO)
    world.logger.propagate = False


    # --- 2. World Creation: A Family, a Threat, and a Tool ---
    root_logger.info("Creating a scene for the dawn of humanity...")

    # The Family
    world.add_cell('human_father', properties={'element_type': 'animal', 'label': 'human', 'diet': 'omnivore', 'hp': 40.0, 'max_hp': 40.0})
    world.add_cell('human_child', properties={'element_type': 'animal', 'label': 'human', 'diet': 'omnivore', 'hp': 10.0, 'max_hp': 10.0})

    # The Threat
    world.add_cell('wolf_1', properties={'element_type': 'animal', 'label': 'wolf', 'diet': 'carnivore', 'hp': 20.0, 'max_hp': 20.0})
    world.add_cell('wolf_2', properties={'element_type': 'animal', 'label': 'wolf', 'diet': 'carnivore', 'hp': 20.0, 'max_hp': 20.0})

    # The Environment & Tools
    world.add_cell('tree_A', properties={'element_type': 'life', 'label': 'tree', 'hp': 100.0, 'max_hp': 100.0})
    world.add_cell('stone_A', properties={'element_type': 'item', 'label': 'stone', 'hp': 1.0, 'max_hp': 1.0})
    world.add_cell('earth_A', properties={'element_type': 'earth', 'hp': 10.0, 'max_hp': 10.0})


    # --- 3. Connections ---
    root_logger.info("Establishing connections...")

    # Family bonds and community
    world.add_connection('human_father', 'human_child', strength=0.9)
    world.add_connection('human_child', 'human_father', strength=0.9)

    # Humans connected to their environment and potential tools
    world.add_connection('human_father', 'tree_A', strength=0.5)
    world.add_connection('human_father', 'stone_A', strength=0.5)
    world.add_connection('human_father', 'earth_A', strength=0.1)
    world.add_connection('human_child', 'tree_A', strength=0.2)

    # Threats are connected to the family
    world.add_connection('wolf_1', 'human_child', strength=0.7)
    world.add_connection('wolf_2', 'human_father', strength=0.6)
    world.add_connection('wolf_2', 'human_child', strength=0.7)
    world.add_connection('human_father', 'wolf_1', strength=0.6) # The father can fight back
    world.add_connection('human_father', 'wolf_2', strength=0.6)

    # Critical missing link: Wolves must also see the father as a target/threat
    world.add_connection('wolf_1', 'human_father', strength=0.6)


    # --- 3.5. Staging the Scene (Manual Positioning) ---
    root_logger.info("Staging the scene by setting manual positions...")
    father_idx = world.id_to_idx.get('human_father')
    child_idx = world.id_to_idx.get('human_child')
    wolf1_idx = world.id_to_idx.get('wolf_1')
    wolf2_idx = world.id_to_idx.get('wolf_2')

    if father_idx is not None: world.positions[father_idx] = [0, 0, 0]
    if child_idx is not None: world.positions[child_idx] = [1, 0, 0] # Close to the father
    if wolf1_idx is not None: world.positions[wolf1_idx] = [5, 2, 0] # Closer
    if wolf2_idx is not None: world.positions[wolf2_idx] = [6, -1, 0] # Closer


    root_logger.info("Initial world state:")
    world.print_world_summary()

    # --- 4. Simulation Loop ---
    num_steps = 20
    root_logger.info(f"Running simulation for {num_steps} steps...")
    for i in range(num_steps):
        print(f"\n--- Running Step {i+1} ---")

        world.run_simulation_step()
        world.print_world_summary()

        # Check for the key events
        if 'wolf_1' in world.id_to_idx and not world.is_alive_mask[world.id_to_idx['wolf_1']]:
             root_logger.info("EVENT: The first wolf has been defeated.")

        calls = mock_wave_mechanics.inject_stimulus.call_args_list
        for call in calls:
            if 'love' in call.args or 'protection' in call.args:
                root_logger.info("SUCCESS: The act of protection has resonated with Elysia's core values!")
                mock_wave_mechanics.inject_stimulus.call_args_list = [] # Clear calls to avoid repeated success messages
            if 'wisdom' in call.args:
                root_logger.info("SUCCESS: The use of a tool has resonated with Elysia's 'wisdom'!")
                mock_wave_mechanics.inject_stimulus.call_args_list = []


    root_logger.info("Genesis simulation complete.")

if __name__ == "__main__":
    main()
