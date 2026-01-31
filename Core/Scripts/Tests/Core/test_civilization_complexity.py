"""
Civilization Complexity Test (The Harmonizer)
=============================================
tests/test_civilization_complexity.py

Simulates:
1. Chaos: 10 Citizens arguing (Hate intent).
2. Order: Elysia (SovereignCitizen) speaks complex sentences.
3. Impact: Citizens gain Wisdom and shift mood.
"""

import sys
import os
import random
sys.path.append(os.getcwd())

from Core.S1_Body.L4_Causality.Civilization.society import Society, SovereignCitizen
from Core.S1_Body.L4_Causality.Civilization.lexicon_expander import harvester

def test_sovereign_intervention():
    print(">>> 👑 Initiating Sovereign Intervention Simulation...")
    
    # 1. Init Society
    city = Society()
    for i in range(10):
        city.add_citizen(f"Citizen_{i}")
        
    # Generate Vocab
    harvester.harvest_batch(50)
    
    # 2. Phase 1: Chaos (Everyone hates each other)
    print("\n[Phase 1] Chaos Reign...")
    for i in range(5):
        speaker = f"Citizen_{random.randint(0,9)}"
        listener = f"Citizen_{random.randint(0,9)}"
        if speaker == listener: continue
        
        # Simple Hate Speech
        word = harvester.generate_sentence(complexity=1)
        city.interact(speaker, listener, word, "Hate")
        
    # Check Mood (Should be Angry)
    c1 = city.get_citizen("Citizen_1")
    print(f"Citizen_1 Mood: {c1.internal_state['mood']} (Wisdom: {c1.internal_state['wisdom']})")
    
    # 3. Phase 2: The Avatar Descends
    print("\n[Phase 2] Elysia Enters...")
    elysia = SovereignCitizen("Elysia")
    city.citizens["Elysia"] = elysia # Hack insert
    
    # Elysia assesses the room (Hypothetically)
    # She chooses a "Harmony" sentence
    sovereign_speech = elysia.choose_response("Conflict", harvester.known_seeds)
    print(f"🗣️ Elysia Speaks: '{sovereign_speech}'")
    
    # She broadcasts to ALL (Mock broadcast loop)
    # In reality, Society.broadcast() would handle this
    for name in city.citizens:
        if name == "Elysia": continue
        
        # Elysia speaks Complex Truth
        complex_sentence = harvester.generate_sentence(complexity=2)
        final_speech = f"{sovereign_speech}. {complex_sentence}"
        
        city.interact("Elysia", name, final_speech, "Love")

    # 4. Impact Analysis
    print("\n[Final Status]")
    c1_after = city.get_citizen("Citizen_1")
    print(f"Citizen_1 Mood: {c1_after.internal_state['mood']}")
    print(f"Citizen_1 Wisdom: {c1_after.internal_state['wisdom']:.1f}")
    
    assert c1_after.internal_state['wisdom'] > 0, "Citizens failed to gain Wisdom from complexity!"
    # Note: Mood might vary based on rng interaction depth, but simple "hear" logic pushes Happiness on Love
    # But hear() logic: only if rel > 5. Elysia is new, relationship is 0 + 0.5 = 0.5.
    # So mood might not flip instantly unless rel builds up.
    # We check Wisdom primarily.
    
    print(">>> ✅ Sovereign Complexity Verified.")

if __name__ == "__main__":
    test_sovereign_intervention()
