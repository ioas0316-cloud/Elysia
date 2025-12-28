"""
Style Learning Simulation
-------------------------
Demonstrates how Elysia can "internalize" expressive capability.

1. Phase 1: Naive State (Elysia speaks simply).
2. Phase 2: Learning (We show her a "good" example).
3. Phase 3: Application (Elysia speaks on a new topic, using the learned style).
"""

import sys
import os
import logging
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Elysia.core_memory import CoreMemory
from Project_Elysia.high_engine.style_learner import StyleLearner
from Project_Elysia.high_engine.utterance_composer import UtteranceComposer
from Project_Elysia.high_engine.intent_engine import IntentBundle

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Fix for Windows Unicode printing
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 70)
    print("🎨 Style Learning Simulation")
    print("=" * 70)

    # Initialize
    core_memory = CoreMemory(file_path=None)
    style_learner = StyleLearner(core_memory)
    composer = UtteranceComposer(core_memory)
    
    # Mock Intent
    intent = IntentBundle(
        target="user",
        emotion="empathetic",
        intent_type="comfort",
        style="conversational",
        relationship="ally",
        law_focus=["+love"]
    )

    # Phase 1: Naive State
    print("\n--- Phase 1: Naive State ---")
    print("Elysia has no learned styles yet.")
    response_1 = composer.compose(intent, base_text="I am here for you.")
    print(f"🤖 Elysia: {response_1}")
    
    # Phase 2: Learning
    print("\n--- Phase 2: Learning ---")
    exemplar = "Even the darkest night will end and the sun will rise. Your pain is but a passing cloud."
    print(f"📖 Reading Exemplar: '{exemplar}'")
    
    learned_pattern = style_learner.learn_from_example(exemplar)
    print(f"💡 Learned Pattern: {learned_pattern}")
    
    # Phase 3: Application
    print("\n--- Phase 3: Application ---")
    print("Elysia speaks on a NEW topic (Hope), applying the learned pattern.")
    
    # New Intent (Hope)
    intent_hope = IntentBundle(
        target="user",
        emotion="hopeful",
        intent_type="encourage",
        style="poetic",
        relationship="guide",
        law_focus=["+creation"]
    )
    
    response_2 = composer.compose(intent_hope, base_text="You can do it.")
    print(f"🤖 Elysia (Evolved): {response_2}")
    
    print("\n" + "=" * 70)
    print("✅ Simulation Complete.")

if __name__ == "__main__":
    main()
