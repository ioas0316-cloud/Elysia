
import sys
import os
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.World.Physics.field_store import universe_field, FieldExcitation
from Core.Soul.lumina_npc import Lumina

def run_aethelgard_demo():
    print("🎬 Starting 'The Harmony of Aethelgard' Demo...")
    
    # 1. Setup Lumina in the Village
    village_center = (0, 0, 0, 0)
    # Excite the world so the Star has something to heat up
    for x in range(-2, 3):
        for z in range(-2, 3):
            universe_field.excite((x, 0, z, 0), FieldExcitation(density_w=10.0))
            
    lumina = Lumina(pos=village_center)
    print(f"📍 {lumina.name} has arrived at the village center.")

    # 2. Cycle of the World
    # -- MORNING --
    print("\n--- [Morning] ---")
    universe_field.star_intensity = 600.0
    universe_field.apply_celestial_harmonic(dt=1.0)
    print(f"LUMINA: {lumina.percieve_and_react()}")
    print(f"WORK:   {lumina.perform_alchemic_task()}")

    # -- AFTERNOON --
    print("\n--- [Afternoon: Wind] ---")
    universe_field.excite((1, 0, 0, 0), FieldExcitation(density_w=100.0))
    print(f"LUMINA: {lumina.percieve_and_react()}")

    # -- NIGHT --
    print("\n--- [Night] ---")
    universe_field.star_intensity = 0.0 
    universe_field.voxels = {} 
    for x in range(-2, 3):
        for z in range(-2, 3):
            universe_field.excite((x, 0, z, 0), FieldExcitation(density_w=5.0))
    universe_field.moon_intensity = 300.0 
    universe_field.apply_celestial_harmonic(dt=1.0) 
    print(f"LUMINA: {lumina.percieve_and_react()}")
    print(f"WORK:   {lumina.perform_alchemic_task()}")

    print("\n--- ✅ Demo Complete ---")

if __name__ == "__main__":
    run_aethelgard_demo()
