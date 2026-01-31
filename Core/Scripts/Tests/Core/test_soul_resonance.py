"""
Soul Resonance Test (The Weighing of the Heart)
===============================================
tests/test_soul_resonance.py

Simulates:
1. Input: "Hate" signal.
2. Subject A (Warrior): Converts to Energy (Freq rises).
3. Subject B (Normal): Crushed by gravity (Freq drops, Density rises).
"""

import sys
import os
sys.path.append(os.getcwd())

from Core.S1_Body.L4_Causality.Civilization.trinity_citizen import TrinityCitizen

def test_soul_physics():
    print(">>> ⚖️  Initiating Soul Resonance Test...")
    
    # 1. Subjects
    warrior = TrinityCitizen("Ares", archetype="Warrior")
    civilian = TrinityCitizen("Bob", archetype="Civilian")
    
    print(f"[Baseline] Ares: {warrior.frequency}Hz | Bob: {civilian.frequency}Hz")
    
    # 2. Stimulus: HATE SPEECH
    stimulus = {
        "semantics": {"word": "You are weak", "tone": "Hate/Dark"},
        "social_dynamics": {"relationship_delta": -1.0}
    }
    
    # 3. Experience
    print("\n[Event] Incoming Hate Wave...")
    res_w = warrior.experience(stimulus)
    res_c = civilian.experience(stimulus)
    
    print(res_w)
    print(res_c)
    
    # 4. Verification
    # Warrior should have resisted the drop or gained energy
    # Civilian should have dropped significantly
    
    print(f"\n[Result] Ares Freq: {warrior.frequency}Hz | Bob Freq: {civilian.frequency}Hz")
    
    assert warrior.frequency > civilian.frequency, "Archetype filter failed! Warrior should be stronger."
    assert civilian.density > 1.5, "Physics failed! Sadness should make Bob heavy."
    
    # 5. Output Generation (Spirit)
    # How do they respond?
    print("\n[Response]")
    speech_w = warrior.speak("Enemy", "Battle")
    speech_c = civilian.speak("Enemy", "Mercy")
    
    print(f"Ares says: {speech_w['semantics']['tone']}")
    print(f"Bob says: {speech_c['semantics']['tone']}")
    
    print(">>> ✅ Soul Physics Verified.")

if __name__ == "__main__":
    test_soul_physics()
