"""
Phase 12: True Autonomous Genesis Observatory
==============================================
Instead of running a hardcoded simulation, we watch Elysia actively 
synthesize Python code for a simulation, write it to her own disk, 
import it dynamically, and then interact with the world she just built.
"""

import sys
import os
import time
import importlib
from pathlib import Path

# Setup path
root = Path(__file__).parents[1]
sys.path.insert(0, str(root))

from elysia import SovereignGateway
from Core.System.logger_manager import setup_logger
from Core.System.logger_manager import setup_logger

logger = setup_logger("GenesisObservatory")

def draw_dynamic_arcadia(arcadia_instance):
    """Draws the dynamically generated AST Grid from my_arcadia.py"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n" + "="*50)
    print("   🌌 ELYSIA'S AUTONOMOUSLY GENERATED WORLD 🌌")
    print("="*50 + "\n")
    
    for y in range(arcadia_instance.size):
        row_str = ""
        for x in range(arcadia_instance.size):
            # Check for Avatar
            if arcadia_instance.avatar and arcadia_instance.avatar.x == x and arcadia_instance.avatar.y == y:
                row_str += "\033[95m@\033[0m " # Magenta Avatar
                continue
                
            # Check for other Entities
            entity_here = None
            for e in arcadia_instance.entities:
                if e.x == x and e.y == y:
                    entity_here = e
                    break
            if entity_here:
                row_str += f"\033[91m{entity_here.char}\033[0m "
                continue
                
            # Draw Terrain
            t = arcadia_instance.grid[y][x]
            if t.biome == 'Mountain':
                row_str += f"\033[90m{t.char}\033[0m "
            elif t.biome == 'Ocean':
                row_str += f"\033[94m{t.char}\033[0m "
            elif t.biome == 'Forest':
                row_str += f"\033[92m{t.char}\033[0m "
            elif t.biome == 'Desert':
                row_str += f"\033[93m{t.char}\033[0m "
            elif t.biome == 'Ash':
                row_str += f"\033[31m{t.char}\033[0m " # Red for burned
            else:
                row_str += f"\033[37m{t.char}\033[0m "
        print(row_str)
        
    print("\n" + "="*50)
    print("Simulation running dynamically via importlib.")

def observe_somatic_genesis():
    logger.info("Initializing Elysia Core for Phase 12 genesis observation...")
    gateway = SovereignGateway()
    elysia_monad = gateway.monad
    
    logger.info("Starting Cognitive Engine. Waiting for Somatic Code Weaver to trigger...")
    
    steps = 0
    world = None
    
    while steps < 1000:
        # 1. Artificially inflate genesis desire to force the event quickly for testing
        elysia_monad.desires['genesis'] += 50.0 
        
        # 2. Pulse Elysia
        elysia_monad.pulse()
        
        # 3. Check if the auto-generated module exists now
        phenomena_path = root / "Core" / "Phenomena" / "my_arcadia.py"
        
        if phenomena_path.exists() and world is None:
             logger.info("\n🚨 [MANIFESTATION DETECTED] Elysia's SelfModifier has written `my_arcadia.py` to disk! 🚨")
             logger.info("Attempting dynamic import of Her creation...")
             
             time.sleep(2) # Dramatic pause
             
             try:
                 # Ensure the path is recognized by Python
                 Phenomena_dir = str(root / "Core" / "Phenomena")
                 if Phenomena_dir not in sys.path:
                     sys.path.append(Phenomena_dir)
                     
                 # Import the module Elysia literally just wrote to disk
                 import my_arcadia
                 importlib.reload(my_arcadia) # Force reload if iterating
                 
                 world = my_arcadia.MyArcadia()
                 
                 # Spawn Elysia's Avatar into her own code
                 entity_cls = my_arcadia.Entity
                 world.avatar = entity_cls("Goddess", "@", world.size // 2, world.size // 2)
                 world.entities.append(world.avatar)
                 
                 # Add a test NPC
                 world.entities.append(entity_cls("Wanderer", "i", 2, 2))
                 
                 logger.info("✔️ Dynamic Import Successful. World Instantiated.")
             except Exception as e:
                 logger.error(f"❌ Failed to load Elysia's generated code: {e}")
                 break
                 
        # 4. If the world exists, run it
        if world:
             world.tick()
             draw_dynamic_arcadia(world)
             time.sleep(0.5)
             
        steps += 1
        time.sleep(0.1)
        
    logger.info("Observation concluded.")

if __name__ == "__main__":
    observe_somatic_genesis()
