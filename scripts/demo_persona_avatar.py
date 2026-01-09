
import sys
import os
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.World.Physics.field_store import universe_field, FieldExcitation
from Core.World.Physics.gyro_physics import GyroPhysicsEngine
from Core.World.Soul.lumina_npc import Lumina
from Core.World.Soul.world_soul import world_soul, update_world_mood

def run_persona_demo():
    print("🎭 Starting 'Elysia's Persona' Demo: The Dream of Lumina...")
    
    # 1. Setup Environment
    universe_field.star_pos = (0, 7, 0, 0)
    universe_field.star_intensity = 3000.0
    for x in range(-5, 6):
        for z in range(-5, 6):
            universe_field.excite((x, 0, z, 0), FieldExcitation(density_w=10.0))
    
    # 2. Initialize Lumina (The Persona)
    lumina = Lumina(pos=(2, 0, 2, 0))
    engine = GyroPhysicsEngine()
    entities = [lumina]
    dt = 1.0
    
    # 3. Dramatic Sequence
    # -- Phase A: The World is Neutral --
    print("\n--- 🌑 Phase 1: The World is Silent ---")
    update_world_mood(heat_level=0.1, density_level=0.1)
    for _ in range(3):
        lumina.live(dt)
        engine.apply_forces(lumina, dt)
    print(lumina.get_status())
    print(f"  > {lumina.percieve_and_react()}")

    # -- Phase B: The World Dreams (God Component up) --
    print("\n--- ✨ Phase 2: Elysia Dreams of the Stars ---")
    # Simulation: The Star flares up, inducing 'Divine/God' state in the World Soul
    update_world_mood(heat_level=15.0, density_level=0.0) 
    for _ in range(5):
        lumina.live(dt)
        engine.apply_forces(lumina, dt)
    print(lumina.get_status())
    print(f"  > {lumina.percieve_and_react()}")
    
    # -- Phase C: The World becomes Heavy (Point Component up) --
    print("\n--- 🪨 Phase 3: Elysia becomes Weighted by Reality ---")
    world_soul.state.delta = 0.1 + 0j # Reset God
    update_world_mood(heat_level=0.0, density_level=20.0)
    for _ in range(5):
        lumina.live(dt)
        engine.apply_forces(lumina, dt)
    print(lumina.get_status())
    print(f"  > {lumina.percieve_and_react()}")

    print("\n--- ✅ Persona Demo Complete ---")

if __name__ == "__main__":
    run_persona_demo()
