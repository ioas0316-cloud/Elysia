
import sys
import os
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.Intelligence.Knowledge.semantic_field import semantic_field
from Core.Intelligence.Knowledge.observer_protocol import observer
from Core.Intelligence.Knowledge.infinite_ingestor import ingestor
from Core.Soul.adventurer import Adventurer, Party
from Core.Intelligence.Logos.logos_engine import get_logos_engine

def run_final_sage_demo():
    print("🧙‍♂️ [FINAL SAGE DEMO] Starting Unified Knowledge Convergence...")
    
    # 0. Clean old memory for a clean demo
    if os.path.exists(semantic_field.save_path):
        os.remove(semantic_field.save_path)
    semantic_field.glossary = {}
    semantic_field.concepts = {}

    # 1. Internal Knowledge Sweep (Convergence)
    print("\n--- 🧹 Phase 1: Internal Domain Convergence ---")
    from scripts.knowledge_convergence_sweep import run_convergence
    run_convergence()
    
    # 2. External Observer (Active Learning)
    print("\n--- 👁️ Phase 2: External Observer Absorption ---")
    wiki_title = "Advanced Magic Systems"
    wiki_content = """
    Vancian Magic: Spells are packaged units that are forgotten after casting. Requires preparation.
    Mana Reservoir: Magic energy drawn from the world soul or external field.
    Backlash: A negative physical feedback when a spell fails due to poor resonance.
    Quintessence: The highest form of magic energy, representing the fifth element of the soul.
    """
    observer.distill_and_ingest(wiki_title, wiki_content)
    
    print(f"\n✅ Persistent Semantic Field now holds {len(semantic_field.glossary)} unified concepts.")

    # 3. Social Resonance (Applying the Unified Knowledge)
    print("\n--- 👥 Phase 3: Lived Wisdom (Multi-NPC Performance) ---")
    kael = Adventurer(name="Kael", archetype="Knight", pos=(0,0,0,0))
    lumina = Adventurer(name="Lumina", archetype="Alchemist", pos=(1,0,1,0))
    party = Party([kael, lumina])
    
    # Kael 'learns' a myth and a magic concept
    kael.ego.adopt_principle("Archetype: Hero", semantic_field.get_concept_pos("Archetype: Hero"))
    kael.ego.adopt_principle("Advanced Magic Systems: Mana Reservoir", semantic_field.get_concept_pos("Advanced Magic Systems: Mana Reservoir"))
    
    print("Social interaction for 3 ticks...")
    for _ in range(3):
        party.update(dt=1.0)
        
    print(f"\n🎬 {kael.name}: \"{kael.speak()}\"")
    print(f"🎬 {lumina.name}: \"{lumina.speak()}\"")
    
    # 4. Final Awareness check
    print("\n--- 🏁 Phase 4: Persistence Check ---")
    # We re-init a NEW semantic field to prove it loads from disk
    from Core.Intelligence.Knowledge.semantic_field import SemanticField
    new_field = SemanticField()
    print(f"📦 Re-loaded persistence count: {len(new_field.glossary)}")
    
    if len(new_field.glossary) > 20:
        print("\n✅ [VERIFIED] Elysia has successfully internalized fragmented, internal, and external knowledge into a persistent, usable Hypercosmos.")

if __name__ == "__main__":
    run_final_sage_demo()
