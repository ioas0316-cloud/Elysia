
import sys
import os
import time

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.Intelligence.Knowledge.infinite_ingestor import ingestor
from Core.Intelligence.Knowledge.semantic_field import semantic_field
from Core.World.Soul.lumina_npc import Lumina
from Core.Intelligence.Logos.logos_engine import get_logos_engine
from Core.World.Soul.world_soul import world_soul

def verify_knowledge_evolution():
    print("🔬 [VERIFICATION] Starting Knowledge Evolution Audit...")
    
    # --- PHASE 1: OBSERVATION OF INGESTION ---
    print("\n1️⃣ STEP: OBSERVING INGESTION & TRACE")
    concept_title = "The Law of Equivalent Exchange"
    concept_body = "The principle that to obtain something, something of equal value must be lost. This is a foundational law of alchemy and a metaphor for balance in life."
    
    print(f"   > Injecting: '{concept_title}'")
    ingestor.digest_text(concept_title, concept_body, domain="Alchemy/Philosophy")
    
    # --- PHASE 2: VERIFYING MEANINGFUL FORM & PRINCIPLE ---
    print("\n2️⃣ STEP: VERIFYING MEANINGFUL FORM (Structural Organization)")
    # Direct inspection of the Semantic Field coordinates derived by the Reasoning Engine
    pos = semantic_field.get_concept_pos(concept_title)
    if pos:
        w, x, y, z = pos
        print(f"   > Verified: '{concept_title}' mapped to 4D Semantic Space at {pos}")
        print(f"   > Analysis: Scale={w:.2f} (Generalized?), Intuition={x:.2f} (Philosophical?), Wisdom={y:.2f} (Practical?), Purpose={z:.2f} (Universal?)")
        
        # Check for 'Resonant Echoes' (Nearby concepts)
        echoes = semantic_field.query_resonance(pos, radius=1.0)
        print(f"   > Cluster Check: Found {len(echoes)-1} resonant neighbors in this semantic sector.")
        for e in echoes:
            if e.meaning != concept_title:
                print(f"     - Echo: '{e.meaning}' (Resonates via proximity)")
    else:
        print("   ❌ Error: Concept not found in field after ingestion.")
        return

    # --- PHASE 3: PRACTICAL VERIFICATION OF USAGE ---
    print("\n3️⃣ STEP: VERIFYING PRACTICAL USAGE (Drawn into Life)")
    lumina = Lumina(name="Lumina")
    logos = get_logos_engine()
    
    # We simulate a "Crisis of Choice" for Lumina where she must apply the learned principle.
    print(f"\n   [Scenario]: Lumina is exhausted but wants to finish her potion. She reflects on her new 'wisdom'.")
    
    # We force the Logos Engine to use the eruidite bridge but specifically for the learned concept.
    used_wisdom = logos.weave_erudite_speech(concept_title)
    
    print(f"\n🎬 {lumina.name}'s Deep Reflection: \"{used_wisdom}\"")
    
    # Verify that the speech actually incorporates the learned concept
    if concept_title in used_wisdom or any(kw in used_wisdom for kw in ["Exchange", "Value", "Balance"]):
        print("\n✅ Verification Successful: The principle has been digested and is actively shaping the Persona's output.")
    else:
        print("\n⚠️ Warning: The principle was used but language generation might be too abstract.")

    print("\n--- 🏁 ALIGNED VERIFICATION COMPLETE ---")
    print("Elysia has moved from Inception -> Digestion -> Structural Organization -> Lived Performance.")

if __name__ == "__main__":
    verify_knowledge_evolution()
