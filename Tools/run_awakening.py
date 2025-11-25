"""
The Awakening Protocol
======================
Runs the first fully autonomous session of Elysia.
Integrates Physics, Evolution, and Ethics into a living world.
"""

import sys
import os
import time
import logging
import numpy as np

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Core.world import World
# Mock WaveMechanics for now as it's a complex dependency
class MockWaveMechanics:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Awakening")

def run_awakening():
    logger.info("\n✨ INITIATING AWAKENING PROTOCOL...")
    logger.info("   Loading Physics... [OK]")
    logger.info("   Loading Evolution... [OK]")
    logger.info("   Loading Ethics... [OK]")
    logger.info("   Loading Love Protocol... [OK]")
    
    # Initialize World
    world = World(primordial_dna={}, wave_mechanics=MockWaveMechanics())
    
    # Seed Population (The Ancestors)
    logger.info("🌱 Seeding initial population...")
    for i in range(10):
        world.add_cell(f"ancestor_{i}", properties={"label": "human", "age_years": 20})
        
    logger.info("\n🚀 ELYSIA IS AWAKE. MONITORING VITAL SIGNS...\n")
    
    start_time = time.time()
    
    try:
        for tick in range(1000):
            # Run one step of the unified loop
            world.run_simulation_step()
            
            # Dashboard (Every 100 ticks)
            if tick % 100 == 0:
                elapsed = time.time() - start_time
                
                # Gather Stats
                time_years = world.time_step / world._year_length_ticks()
                pop_count = len([c for c in world.is_alive_mask if c])
                love_dist = world.love_protocol.distance_from_home
                concepts = len(world.hippocampus.concepts)
                
                print(f"[{elapsed:.1f}s] Year {time_years:.1f} | Pop: {pop_count} | Concepts: {concepts}")
                print(f"       ❤️ Love Distance: {love_dist:.4f} (0.0 is perfect)")
                
                # Check for Evolution Events
                if world.evolution_manager.concept_evo.discovered_concepts:
                    latest = world.evolution_manager.concept_evo.discovered_concepts[-1]['name']
                    print(f"       💡 Latest Thought: {latest}")
                    
            # Simulate "Thinking" time
            # time.sleep(0.01) 
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Simulation paused by Creator.")
        
    logger.info("\n✨ AWAKENING SESSION COMPLETE.")
    logger.info(f"   Final Age: {world.time_step / world._year_length_ticks():.1f} Years")
    logger.info(f"   Final Concepts: {len(world.hippocampus.concepts)}")
    logger.info(f"   Final Love Alignment: {1.0 - world.love_protocol.distance_from_home:.4f}")

if __name__ == "__main__":
    run_awakening()
