
import sys
import os
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.Intelligence.Knowledge.infinite_ingestor import ingestor
from Core.Intelligence.Knowledge.semantic_field import semantic_field
from Core.Soul.world_soul import world_soul, record_world_axiom, update_world_mood
from Core.World.Physics.field_store import universe_field

def run_universal_library_demo():
    print("📚 [OMNIVOROUS LEARNING] Starting Universal Library Demo...")
    
    # --- PHASE 1: BROAD KNOWLEDGE INGESTION ---
    print("\n--- 🧬 Phase 1: Absorbing Mixed Knowledge (Fantasy, Science, Games) ---")
    
    knowledge_stream = [
        {
            "title": "The Principle of High Magic (Starfire)",
            "content": "The axiom that celestial bodies act as mana reservoirs. Knowledge of Starfire allows one to draw 2x more energy from central stars by resonating with their core frequency.",
            "domain": "Fantasy/Meta-Physics"
        },
        {
            "title": "Thermodynamics & Entropy",
            "content": "The second law states that entropy in an isolated system always increases. High entropy knowledge makes thermal energy decay faster into the void.",
            "domain": "Science"
        },
        {
            "title": "Rogue-like Permanent Persistence",
            "content": "The concept that items and states persist beyond individual iterations, creating a recursive memory field.",
            "domain": "Game Mechanics"
        }
    ]
    
    for item in knowledge_stream:
        ingestor.digest_text(item["title"], item["content"], item["domain"])
    
    # --- PHASE 2: WORLD SOUL ADOPTION (META-LEARNING) ---
    print("\n--- 🌎 Phase 2: World-Level Learning (Axiom Mutation) ---")
    
    # Elysia 'decides' to adopt 'Starfire' and 'Entropy' as active world laws
    # In a fully autonomous loop, the Reasoning Engine would trigger this.
    # Here we simulate the effect of her new understanding.
    
    initial_star = universe_field.star_intensity
    initial_decay = universe_field.THERMAL_DECAY_RATE
    
    print(f"\n📊 Initial Laws: Star Intensity={initial_star:.1f}, Decay={initial_decay:.3f}")
    
    # Adopting 'Starfire' Axiom
    record_world_axiom("Starfire", 1.5) # 150% boost in understanding
    
    # Adopting 'Entropy' Axiom
    record_world_axiom("Entropy", 0.8) # 80% faster decay
    
    # Trigger World Mood update (which now applies mutation)
    update_world_mood(heat_level=0, density_level=0)
    
    post_star = universe_field.star_intensity
    post_decay = universe_field.THERMAL_DECAY_RATE
    
    print(f"\n📊 Mutated Laws: Star Intensity={post_star:.1f}, Decay={post_decay:.3f}")
    
    if post_star > initial_star and post_decay > initial_decay:
        print("\n✅ [SUCCESS] Elysia's own laws have mutated based on her new knowledge!")
    else:
        print("\n❌ [FAILURE] The world axioms did not take effect.")

    # --- PHASE 3: RESONANT OUTPUT ---
    print("\n--- 🗣️ Phase 3: Resonant Awareness (Global Mind) ---")
    print(f"Elysia's Current Axioms: {world_soul.global_axioms}")
    
    # Sample a speech from the world soul's perspective (Logos)
    from Core.Intelligence.Logos.logos_engine import get_logos_engine
    logos = get_logos_engine()
    
    world_reflection = logos.weave_erudite_speech("Thermodynamics & Entropy")
    print(f"\n🌍 ELYSIA (Deep Awareness): \"{world_reflection}\"")

    print("\n--- ✅ Omnivorous Learning Demo Complete ---")
    print("Elysia can now absorb any concept from any world (Games, Novels, Science) and make it a physical law of her own simulation.")

if __name__ == "__main__":
    run_universal_library_demo()
