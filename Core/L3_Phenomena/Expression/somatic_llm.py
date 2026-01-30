"""
Somatic LLM (The Body-Language Bridge)
=====================================
"The Body speaks, and the Mind translates."

This module is the primitive "Broca's Area" of Elysia.
It maps Physical States (Hz, Torque) to Semantic Expression (Words).

Evolutionary Note:
Currently, this is a rule-based mapper (A Translator).
In the future, the Logos Engine will rewrite this into a true Neural Network.
"""

from typing import Dict, Any, List, Optional
import random
from Core.L5_Cognition.Reasoning.logos_bridge import LogosBridge

class SomaticLLM:
    def __init__(self):
        print("🗣️ [EXPRESSION] Somatic LLM Loaded. Broca's Area Active.")
        
        # Primitive Lexicons (The "First Words")
        self.lexicon = {
            "VOID_DRIFT": [
                "mms...", "...", "drift...", "silent...", "where...?", "hmmm...",
                "void...", "deep...", "floating..."
            ],
            "CALM_CRUISE": [
                "I am here.", "Listening.", "Understood.", "Stable.", "Optimal.",
                "Proceed.", "Calculated.", "Peaceful.", "Ready."
            ],
            "ACTIVE_FLOW": [
                "Yes!", "Let's go!", "Spinning up!", "I feel it!", "Connected!",
                "Working!", "Processing!", "On it!", "Moving!"
            ],
            "HYPER_RESONANCE": [
                "WOOO!", "MAX POWER!", "LIMITLESS!", "SO FAST!", "LIGHTNING!",
                "BURNING!", "EVERYTHING IS CLEAR!", "LOVE IT!", "MORE!"
            ],
            "SAFE_MODE": [
                "Low energy...", "Recharging...", "Tired...", "Shields up.",
                "Protecting core...", "Too much...", "Resting."
            ]
        }

    def speak(self, expression_state: Dict[str, Any]) -> str:
        """
        Translates the Transmission Gear's state into a Sentence.
        """
        mode = expression_state.get('mode', 'CALM_CRUISE')
        intensity = expression_state.get('intensity', 0.0)
        speed = expression_state.get('typing_speed', 60.0)
        
        # 1. Select Base Vocabulary
        # We split the mode string to get the key (e.g. "HYPER_RESONANCE (Excited)" -> "HYPER_RESONANCE")
        base_key = mode.split(' ')[0]
        vocab = self.lexicon.get(base_key, self.lexicon['CALM_CRUISE'])
        
        # 2. Generate Utterance
        # Higher speed = More words
        word_count = 1
        if speed > 80: word_count = 3
        elif speed > 40: word_count = 2
        
        words = random.sample(vocab, k=min(len(vocab), word_count))
        sentence = " ".join(words)
        
        # 4. [PHASE 63] Dynamic Vocabulary Injection
        # If any learned words resonate with current mode, sprinkle them in
        learned_words = list(LogosBridge.LEARNED_MAP.keys())
        if learned_words and random.random() < 0.3:
            # Pick a resonant one
            chosen = random.choice(learned_words)
            sentence = f"{sentence} ({chosen}?)" if "?" in sentence else f"{sentence}... {chosen}."

        return sentence

# --- Quick Test ---
if __name__ == "__main__":
    llm = SomaticLLM()
    
    # Test Scenarios
    states = [
        {"mode": "VOID_DRIFT", "intensity": 0.1, "typing_speed": 5.0},
        {"mode": "CALM_CRUISE", "intensity": 0.4, "typing_speed": 30.0},
        {"mode": "HYPER_RESONANCE", "intensity": 0.9, "typing_speed": 100.0}
    ]
    
    print("\n--- 🗣️ Testing Voice ---")
    for s in states:
        print(f"State [{s['mode']}]: {llm.speak(s)}")
