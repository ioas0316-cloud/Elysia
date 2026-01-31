"""
Civilization Genesis Test (The First Conversation)
==================================================
tests/test_civilization_conversation.py

Simulates:
1. Two Monads (Adam, Eve) exchanging Words.
2. Context (Time) altering the meaning of "Hello".
3. Intent (Sarcasm/Love) altering the relationship impact.
"""

import sys
import os
sys.path.append(os.getcwd())

from Core.L7_Spirit.M1_Monad.seed_factory import alchemy
from typing import Dict, Any

def test_conversation():
    print(">>> 🗣️ Initiating Civilization Genesis (The Conversation)...")
    
    # 1. Crystallize a Word-Monad
    word_monad = alchemy.crystallize("Hello")
    print(f"Word Created: {word_monad}")
    
    # 2. Scene 1: Morning Greeting (Time = 08:00)
    # Adam says "Hello" to Eve with Love.
    context_morning = {
        "time": 8.0, 
        "speaker": "Adam", 
        "listener": "Eve"
    }
    intent_love = {
        "emotional_texture": "Love/Warm",
        "focus_topic": "Greeting"
    }
    
    reality_morning = word_monad.observe(intent_love, context_morning)["manifestation"]
    
    sem_m = reality_morning.get("semantics", {})
    soc_m = reality_morning.get("social_dynamics", {})
    
    print(f"\n[Scene 1: Morning Greeting]")
    print(f" - Word: '{sem_m.get('word')}'")
    print(f" - Nuance: {sem_m.get('context_nuance')} (Time 8:00)")
    print(f" - Tone: {sem_m.get('tone')}")
    print(f" - Social Impact: {soc_m.get('interaction_type')} (Delta: {soc_m.get('relationship_delta')})")
    
    assert sem_m.get("context_nuance") == "Fresh/Morning", "Linguistics Failed to detect Morning!"
    assert soc_m.get("relationship_delta") > 0, "Sociology Failed to Bond!"
    
    # 3. Scene 2: Late Night Argument (Time = 02:00)
    # Eve says "Hello" (Sarcastic) to Adam.
    context_night = {
        "time": 2.0, # 2 AM
        "speaker": "Eve", 
        "listener": "Adam"
    }
    intent_sarcasm = {
        "emotional_texture": "Sarcasm/Cold",
        "focus_topic": "Mockery"
    }
    
    reality_night = word_monad.observe(intent_sarcasm, context_night)["manifestation"]
    
    sem_n = reality_night.get("semantics", {})
    soc_n = reality_night.get("social_dynamics", {})
    
    print(f"\n[Scene 2: Night Argument]")
    print(f" - Word: '{sem_n.get('word')}'")
    print(f" - Nuance: {sem_n.get('context_nuance')} (Time 2:00)")
    print(f" - Tone: {sem_n.get('tone')}")
    print(f" - Social Impact: {soc_n.get('interaction_type')} (Delta: {soc_n.get('relationship_delta')})")
    
    assert sem_n.get("context_nuance") == "Deep/Night", "Linguistics Failed to detect Night!"
    assert soc_n.get("relationship_delta") < 0, "Sociology Failed to detect Sarcasm!"
    
    print(">>> ✅ Civilization Conversation Verified.")

if __name__ == "__main__":
    test_conversation()
