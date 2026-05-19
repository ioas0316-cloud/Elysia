"""
빠른 검증: 1000 사이클로 개념 증명
"""

import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.universe_evolution import UniverseEvolutionEngine
from Core.FoundationLayer.Foundation.spiderweb import Spiderweb
from Core.FoundationLayer.Foundation.core.world import World
from Core.FoundationLayer.Foundation.wave_mechanics import WaveMechanics
from Project_Elysia.core_memory import Experience
from tools.kg_manager import KGManager
from datetime import datetime

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    print("🚀 Quick Universe Evolution Test (1000 cycles)")
    
    # Universe
    kg_manager = KGManager()
    wave_mechanics = WaveMechanics(kg_manager)
    world = World(
        primordial_dna={"instinct": "connect", "resonance": "love"},
        wave_mechanics=wave_mechanics
    )
    
    spiderweb = Spiderweb()
    engine = UniverseEvolutionEngine(world, spiderweb)
    
    # Experiences
    exps = [
        Experience(datetime.now().isoformat() + f"_{i}", f"Test experience {i}", "episode")
        for i in range(3)
    ]
    
    engine.spawn_experience_universe(exps)
    print(f"✓ Spawned {world.is_alive_mask.sum()} particles")
    
    # Evolve
    print("⚡ Evolving...")
    engine.evolve(cycles=1000, extract_interval=200)
    
    # Results
    print(f"\n📊 Results:")
    print(f"  Nodes: {spiderweb.graph.number_of_nodes()}")
    print(f"  Edges: {spiderweb.graph.number_of_edges()}")
    print(f"  Alive: {world.is_alive_mask.sum()}")
    print(f"  Max coherence: {world.coherence_field.max():.3f}")
    print("\n✅ Quick test complete!")

if __name__ == "__main__":
    main()
