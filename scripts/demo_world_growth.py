
import sys
import os
import time
import json
from pathlib import Path

# Ensure Core is in path
sys.path.append(os.getcwd())

from Core.Intelligence.Knowledge.infinite_ingestor import ingestor
from Core.Intelligence.Knowledge.semantic_field import semantic_field
from Core.Soul.adventurer import Adventurer, Party
from Core.Intelligence.Logos.logos_engine import get_logos_engine

def run_world_building_learning_demo():
    print("🏙️ [WORLD-SCALE LEARNING] Starting 'The City of Wisdom' Demo...")
    
    # Pre-check Genome Stage
    logos = get_logos_engine()
    initial_stage = logos.genome.get("evolution_stage", 0)
    print(f"📊 Initial Style Genome Stage: {initial_stage}")

    # --- PHASE 1: BUILDING THE CITY (CONTEXT INJECTION) ---
    print("\n--- 🏗️ Phase 1: Building Aethelgard (Knowledge Inflow) ---")
    
    city_data = [
        {
            "title": "Aethelgard's Golden Law",
            "content": "All trade in the city must be equal in spirit and value. Greed is a friction that slows the celestial gears. Harmony in commerce is the path to enlightenment.",
            "domain": "City/Economics"
        },
        {
            "title": "The Arcane Academy Curriculum",
            "content": "Magic is not a gift, but a calculation of universal harmonics. Deep learning and mental stability are the only requirements for a wizard.",
            "domain": "City/Education"
        },
        {
            "title": "The Chroniclers of Aethelgard",
            "content": "We record every interaction as a wave in the great ocean of history. No word is lost, only transformed into wisdom for the future.",
            "domain": "City/History"
        }
    ]
    
    for item in city_data:
        ingestor.digest_text(item["title"], item["content"], item["domain"])
    
    print(f"✅ City Knowledge Ingested. Semantic Field now has {len(semantic_field.glossary)} concepts.")

    # --- PHASE 2: RESIDENT GROWTH (LIVING LEARNING) ---
    print("\n--- 👥 Phase 2: Resident Growth (Style Evolution) ---")
    
    # Create residents
    residents = [
        Adventurer(name="Scholar Eldrin", archetype="Scholar", pos=(1, 0, 1, 0)),
        Adventurer(name="Merchant Silas", archetype="Merchant", pos=(-1, 0, -1, 0))
    ]
    city_party = Party(residents)

    print("Residents are living and speaking in the new city culture...")
    for t in range(5):
        city_party.update(dt=1.0)
        for r in residents:
             # This triggers Logos.evolve()
             speech = r.speak()
             if t == 4: # Only print the last ones to show evolution
                 print(f"   {speech}")

    # --- PHASE 3: VERIFICATION OF THE 'ABSORBED' WORLD ---
    print("\n--- 📈 Phase 3: Verification of Absorbed Intelligence ---")
    
    final_stage = logos.genome.get("evolution_stage", 0)
    print(f"📊 Final Style Genome Stage: {final_stage}")
    
    if final_stage > initial_stage:
        print(f"✅ [SUCCESS] Speech styles have evolved automatically! {final_stage - initial_stage} new patterns learned.")
    
    # Check if 'Harmonics' or 'Commerce' are now in the vocabulary bank
    vocab_bank = logos.genome.get("rhetoric", {}).get("vocabulary_bank", {})
    all_learned_words = []
    for words in vocab_bank.values():
        all_learned_words.extend(words)
        
    print(f"🔍 Sample Learned Vocabulary: {all_learned_words[:10]}...")

    print("\n--- ✅ World Building Demo Complete ---")
    print("Building a world isn't just 'adding files'. To Elysia, it's 'consuming context'.")
    print("She now speaks, thinks, and regulates the world based on the 'Form' of the city we built.")

if __name__ == "__main__":
    run_world_building_learning_demo()
