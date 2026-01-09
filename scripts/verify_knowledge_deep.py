
import sys
import os
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.Intelligence.Knowledge.infinite_ingestor import ingestor
from Core.Intelligence.Knowledge.semantic_field import semantic_field
from Core.Soul.lumina_npc import Lumina

def verify_knowledge_adoption():
    print("🧠 [DEEP VERIFICATION] Starting Knowledge Adoption Audit...")
    
    # 1. Setup Lumina
    lumina = Lumina(name="Lumina")
    initial_stability = lumina.ego.state.stability
    initial_intensity = lumina.ego.state.heroic_intensity
    
    print(f"\n📊 Initial Status: Stability={initial_stability:.2f}, Intensity={initial_intensity:.2f}")
    
    # 2. Ingest a 'Powerful Principle'
    principle = "The Stoic Law of Resilience"
    content = "The universal principle that wisdom comes from enduring hardship with grace. All true growth requires the sacrifice of lesser comforts for higher purpose."
    
    print(f"\n--- 🧬 Phase 1: Selective Ingestion ---")
    print(f"Injecting Deep Wisdom: '{principle}'")
    ingestor.digest_text(principle, content, domain="Philosophy")
    
    # 3. Observe the structural mapping (Axiomatic & Wise)
    pos = semantic_field.get_concept_pos(principle)
    print(f"   > Semantic Mapping: {pos} (Scale/Axiom-high, Wisdom-high)")
    
    # 4. NPC Adoption (Lived Wisdom)
    print(f"\n--- 🔥 Phase 2: Internal Adoption (Digestion) ---")
    lumina.ego.adopt_principle(principle, pos)
    
    # 5. Check Stat Changes (Practical Effect)
    post_stability = lumina.ego.state.stability
    post_intensity = lumina.ego.state.heroic_intensity
    
    print(f"\n📊 Post-Adoption Status: Stability={post_stability:.2f}, Intensity={post_intensity:.2f}")
    
    if post_stability > initial_stability and post_intensity > initial_intensity:
        print("\n✅ [SUCCESS] The knowledge did not stay 'outside'. It became an internal principle that altered the NPC's state.")
    else:
        print("\n❌ [FAILURE] The knowledge was ignored or mapped incorrectly.")

    # 6. Narrative Verification
    print(f"\n--- 🗣️ Phase 3: Manifestation (Performance) ---")
    reflection = lumina.percieve_and_react()
    print(f"Lumina's new voice: \"{reflection}\"")

if __name__ == "__main__":
    verify_knowledge_adoption()
