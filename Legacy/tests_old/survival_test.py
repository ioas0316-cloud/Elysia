"""입자 생존 테스트 - peaceful_mode로"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.universe_evolution import UniverseEvolutionEngine
from Core.Foundation.spiderweb import Spiderweb
from Core.Foundation.core.world import World
from Core.Foundation.wave_mechanics import WaveMechanics
from Project_Elysia.core_memory import Experience
from tools.kg_manager import KGManager
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# World with peaceful mode
kg = KGManager()
wave = WaveMechanics(kg)
world = World({"instinct": "survive"}, wave)

# CRITICAL: Peaceful mode - no death!
world.peaceful_mode = True
world.macro_food_model_enabled = True

spider = Spiderweb()
engine = UniverseEvolutionEngine(world, spider)

# Spawn
exps = [Experience(datetime.now().isoformat() + f"_{i}", f"Experience {i}", "episode") for i in range(5)]
engine.spawn_experience_universe(exps)

print(f"Initial alive: {world.is_alive_mask.sum()}")
print(f"Initial HP: {world.hp[world.is_alive_mask].mean():.0f}")

# Evolve
for i in range(5):
    world.run_simulation_step()
    alive = world.is_alive_mask.sum()
    avg_hp = world.hp[world.is_alive_mask].mean() if alive > 0 else 0
    print(f"Step {i+1}: Alive={alive}, HP={avg_hp:.0f}")

print(f"\n✅ Survival test: {world.is_alive_mask.sum()}/5 particles alive after 5 steps")
