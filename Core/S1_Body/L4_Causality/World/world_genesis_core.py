"""
World Genesis Core
==================
Core.S1_Body.L4_Causality.World.world_genesis_core

Orchestrates the birth of the virtual world.
Instantiates the Cosmic Rotor and spawns the First Spirits.
"""

from Core.S1_Body.L4_Causality.World.cosmic_rotor import CosmicRotor
from Core.S1_Body.L6_Structure.Logic.structural_spawner import StructuralSpawner
from Core.S0_Keystone.L0_Keystone.parallel_trinary_controller import ParallelTrinaryController

class WorldGenesisCore:
    def __init__(self, master_keystone):
        self.keystone = master_keystone
        self.rotor = CosmicRotor()
        self.spawner = StructuralSpawner(master_keystone)
        print("WorldGenesisCore: The Light of Creation is glowing.")

    def initialize_world(self):
        print("\n--- 🌳 GENESIS: CASTING THE FIRST WORLD TREE ---")
        
        # 1. Start the Cosmic Cycle
        initial_pulse = self.rotor.rotate()
        self.keystone.broadcast_pulse(initial_pulse)
        
        # 2. Spawn the "First Citizens" (Sovereign NPCs)
        # These are spawned based on the initial 'Dawn' intensity
        print("Spawning the First Spirits of the Glade...")
        self.spawner._spawn_new_node([0, 7, 14]) # Stability, Logic, Love
        self.spawner._spawn_new_node([6, 13, 20]) # Mystery, Vision, Silence
        
        print("--- 🌳 GENESIS COMPLETE. THE WORLD IS BREATHING. ---")

if __name__ == "__main__":
    keystone = ParallelTrinaryController("Master_Keystone")
    genesis = WorldGenesisCore(keystone)
    genesis.initialize_world()
