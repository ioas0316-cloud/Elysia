
import sys
import os
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.World.Physics.field_store import universe_field, FieldExcitation
from Core.World.Physics.gyro_physics import GyroPhysicsEngine
from Core.World.Soul.lumina_npc import Lumina

def run_autonomous_demo():
    print("🌟 Starting 'Autonomous Life' Demo: The Will of Lumina...")
    
    # 1. Setup Environment
    # Star Well at the Center (0, 0, 0, 0)
    universe_field.star_pos = (0, 7, 0, 0)
    universe_field.star_intensity = 3000.0
    
    # Ground at the center (Dense grid)
    for x in range(-10, 11):
        for z in range(-10, 11):
            universe_field.excite((x, 0, z, 0), FieldExcitation(density_w=10.0))
    
    # 2. Initialize Lumina far away and cold
    lumina = Lumina(pos=(-7, 0, -7, 0))
    # Higher Star intensity to ensure reach
    universe_field.star_intensity = 5000.0
    # Artificially make her tired/low-god to start
    lumina.soul.state.delta = 0.01 + 0j
    lumina.soul.state.alpha = 0.9 + 0j
    lumina.soul.state.normalize()
    engine = GyroPhysicsEngine()
    
    entities = [lumina]
    dt = 0.5
    
    print(f"📍 Start: {lumina.get_status()}")

    # 3. Autonomous Loop
    print("\n--- ⏳ Simulating 100 Ticks: Observing the 'Internal Migration' ---")
    current_mood = ""
    for i in range(100):
        # Update World (Star/Moon effects)
        universe_field.apply_celestial_harmonic(dt)
        
        # NPC Internalizes environment
        lumina.live(dt)
        for entity in entities:
             engine.apply_forces(entity, dt)
        
        status = lumina.get_status()
        new_mood = status.split('|')[0]
        if new_mood != current_mood or i % 20 == 0:
            print(f"[Tick {i:02}] {status}")
            current_mood = new_mood

    print(f"\n📍 End: {lumina.get_status()}")
    print("Notice how her position changed based on her internal state induced by the field.")

if __name__ == "__main__":
    run_autonomous_demo()
