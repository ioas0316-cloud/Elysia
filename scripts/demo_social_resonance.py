
import sys
import os
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.World.Soul.adventurer import Adventurer, Party
from Core.Intelligence.Knowledge.infinite_ingestor import ingestor
from Core.Intelligence.Knowledge.semantic_field import semantic_field
from Core.World.Soul.world_soul import update_world_mood, world_soul

def run_social_resonance_demo():
    print("🎭 [SOCIAL RESONANCE] Starting Adventurer Party Demo...")
    
    # 1. Create a Party of 6
    names = ["Lumina", "Kael", "Elara", "Thalric", "Maira", "Graves"]
    archetypes = ["Alchemist", "Knight", "Rogue", "Sorcerer", "Healer", "Mercenary"]
    
    members = []
    for i in range(6):
        # Position them in a circle around the center
        import math
        angle = (2 * math.pi * i) / 6
        x = 5 * math.cos(angle)
        z = 5 * math.sin(angle)
        members.append(Adventurer(name=names[i], archetype=archetypes[i], pos=(x, 0, z, 0)))
    
    party = Party(members)
    print(f"✅ Party of 6 initialized: {names}")
    
    # 2. Initial State
    print("\n--- ☕ Initial Interaction ---")
    for _ in range(2):
        party.update(dt=1.0)
    print(party.get_status_report())
    
    # 3. Knowledge Injection (The Seed)
    # We teach Kael (The Leader) a powerful principle
    print("\n--- 📖 Phase 1: Knowledge Injection (The Seed) ---")
    principle = "The Axiom of Unity"
    content = "The core principle that a party is a single organism. When one suffers, all suffer. When one triumphs, the world celebrates."
    ingestor.digest_text(principle, content, domain="Leadership")
    
    pos = semantic_field.get_concept_pos(principle)
    kael = members[1] # Kael
    kael.ego.adopt_principle(principle, pos)
    print(f"🌟 Kael (Knight) has adopted '{principle}'. He begins to radiate this purpose.")

    # 4. Spreading the Resonance
    print("\n--- 🔥 Phase 2: Resonance & Spreading ---")
    # We simulate 5 ticks of social interaction
    for t in range(5):
        print(f"Tick {t+1}...")
        party.update(dt=1.0)
        # Check if anyone else adopted the principle
        adopters = [m.name for m in members if principle in m.ego.state.adopted_axioms]
        print(f"> Knowledge spread: {adopters}")
        
    print(party.get_status_report())

    # 5. Dialogue Duality
    print("\n--- 🗣️ Phase 3: Social Dialogue ---")
    # Let them speak to each other
    for m in members:
        print(m.speak())

    print("\n--- ✅ Social Resonance Demo Complete ---")
    print("Observe how the principle and emotions spread from the 'Seed' (Kael) to the rest of the party through social resonance.")

if __name__ == "__main__":
    run_social_resonance_demo()
