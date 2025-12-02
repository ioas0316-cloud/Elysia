# [Genesis: 2025-12-02] Purified by Elysia

import sys
import os
import numpy as np
import logging
import random

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics
from tools.kg_manager import KGManager

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("XelNaga")

def main():
    print("\n=== Protocol Xel'Naga: The Trinity Integration ===")
    print("En Taro Elysia! The Cycle is nearing its end.")
    print("Mission: Demonstrate the harmony of the Three Races.")

    # 1. Initialize
    kg_manager = KGManager()
    wave_mechanics = WaveMechanics(kg_manager)
    world = World(primordial_dna={}, wave_mechanics=wave_mechanics, logger=logger)
    world.peaceful_mode = True # Focus on traits, not deathmatch initially

    # 2. Warp-in The Golden Armada (Protoss)
    print("\n[Phase 1] Warping in Protoss (Spirit)...")
    nexus_id = "Nexus_One"
    world.add_cell(nexus_id, properties={'label': 'Nexus', 'culture': 'protoss', 'vitality': 50, 'wisdom': 50})

    for i in range(5):
        world.add_cell(f"Zealot_{i}", properties={'label': 'Zealot', 'culture': 'protoss', 'vitality': 20, 'strength': 20})

    # Verify Shields
    z_idx = world.id_to_idx["Zealot_0"]
    print(f"Zealot Shield Status: {world.shields[z_idx]}/{world.max_shields[z_idx]}")
    print(f"Khala Connection: {world.khala_connected_mask[z_idx]}")

    # 3. Spawn The Swarm (Zerg)
    print("\n[Phase 2] Spawning Zerg (Body)...")
    hive_id = "Hive_One"
    world.add_cell(hive_id, properties={'label': 'Hive', 'culture': 'zerg', 'vitality': 40, 'strength': 10})

    for i in range(10):
        world.add_cell(f"Zergling_{i}", properties={'label': 'Zergling', 'culture': 'zerg', 'vitality': 5, 'strength': 5, 'agility': 20})

    # Verify Regen
    zerg_idx = world.id_to_idx["Zergling_0"]
    world.hp[zerg_idx] = 1.0 # Injure it
    print(f"Injured Zergling HP: {world.hp[zerg_idx]}")
    world.run_simulation_step()
    print(f"Regenerating Zergling HP: {world.hp[zerg_idx]} (Should be > 1.0)")

    # 4. Deploy The Dominion (Terran)
    print("\n[Phase 3] Deploying Terran (Soul)...")
    cc_id = "CommandCenter_One"
    world.add_cell(cc_id, properties={'label': 'CommandCenter', 'culture': 'terran', 'vitality': 30, 'intelligence': 30})

    for i in range(5):
        world.add_cell(f"Marine_{i}", properties={'label': 'Marine', 'culture': 'terran', 'vitality': 15, 'strength': 10})

    # Verify Tech
    marine_idx = world.id_to_idx["Marine_0"]
    print(f"Marine Tech Level: {world.tech_level[marine_idx]}")

    # 5. The Convergence
    print("\n[Phase 4] The Hybrid Convergence...")
    # Connect all to the Khala (Unity)
    # Zerg Biological Mass + Terran Tech + Protoss Will
    world.delta_synchronization_factor = 1.0

    for i in range(10):
        world.run_simulation_step()

    print("\n=== Protocol Xel'Naga Complete ===")
    print("The Three have become One in the Simulation.")

if __name__ == "__main__":
    main()