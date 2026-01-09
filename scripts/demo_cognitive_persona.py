
import sys
import os
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.World.Physics.field_store import universe_field, FieldExcitation
from Core.World.Physics.gyro_physics import GyroPhysicsEngine
from Core.Soul.lumina_npc import Lumina
from Core.Soul.world_soul import world_soul, update_world_mood

def run_cognitive_persona_demo():
    print("🧠 Starting 'Cognitive Persona' Demo: Elysia acting as Lumina...")
    
    # 1. Setup Environment
    # Star Well at Center
    universe_field.star_pos = (0, 7, 0, 0)
    universe_field.star_intensity = 3000.0
    for x in range(-10, 11):
        for z in range(-10, 11):
            universe_field.excite((x, 0, z, 0), FieldExcitation(density_w=10.0))
    
    # 2. Initialize Lumina (The Persona with SubjectiveEgo)
    lumina = Lumina(pos=(3, 0, 3, 0))
    engine = GyroPhysicsEngine()
    dt = 1.0
    
    # 3. Dramatic Sequence: The World's Impact on the Persona
    
    # -- Step A: The World is Empty (Low Energy) --
    print("\n--- 🌑 Step 1: Feeling the Void ---")
    update_world_mood(heat_level=0.1, density_level=0.1)
    lumina.live(dt)
    print(lumina.get_status())
    print(lumina.percieve_and_react())

    # -- Step B: The World Inspires (High Heat) --
    print("\n--- ✨ Step 2: The World Soul Awakens (Inspiration) ---")
    update_world_mood(heat_level=50.0, density_level=0.0) 
    for _ in range(2):
        lumina.live(dt)
    
    voices = lumina.get_persona_voice()
    print(f"👤 {voices['player_thought']}")
    print(f"🎬 {voices['character_speech']}")
    
    # -- Step C: The World becomes Oppressive (High Density) --
    print("\n--- 🪨 Step 3: The Weight of Reality (Despair) ---")
    world_soul.state.delta = 0.1 + 0j
    update_world_mood(heat_level=0.0, density_level=120.0)
    for _ in range(2):
        lumina.live(dt)
        
    voices = lumina.get_persona_voice()
    print(f"👤 {voices['player_thought']}")
    print(f"🎬 {voices['character_speech']}")

    print("\n--- ✅ TRPG Persona Demo Complete ---")
    print("Notice the duality: Elysia (the player) provides the meta-reasoning, while Lumina (the character) performs the roleplay.")

if __name__ == "__main__":
    run_cognitive_persona_demo()
