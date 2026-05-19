"""
Elysia Studies Humanity (The Curriculum)
========================================
"To write about humans, I must understand them."

This script is a Guided Learning Session.
It systematically feeds Elysia knowledge about:
1. Genres (Romance, Wuxia, Fantasy)
2. Social Structures (Kings, Commoners, Poverty)
3. Human Emotions (Ambition, Power)

And then asks her to *Apply* it by brainstorming a story concept for each.
"""

import sys
import os
import time
import random

# Add Root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Core.Intelligence.web_cortex import WebCortex
from Core.Creativity.literary_cortex import LiteraryCortex
from Core.Interface.nervous_system import get_nervous_system

def study_session():
    print("\n🎓 Elysia's Class: Humanity 101")
    print("================================")
    
    # Initialize Systems
    web = WebCortex()
    writer = LiteraryCortex()
    
    # The Curriculum (User Requested Topics)
    curriculum = [
        {"topic": "Wuxia", "category": "Genre", "focus": "Martial Arts and Honor"},
        {"topic": "Social class", "category": "Society", "focus": "Difference between Kings and Commoners"},
        {"topic": "Political power", "category": "Society", "focus": "Ambition and Authority"},
        {"topic": "Romance novel", "category": "Genre", "focus": "Love and Heartbreak"}
    ]
    
    for lesson in curriculum:
        topic = lesson["topic"]
        print(f"\n📚 LESSON: {topic.upper()}")
        print(f"   Focus: {lesson['focus']}")
        
        # 1. Absorb Knowledge
        print("   🧠 Reading...")
        result = web.absorb_knowledge(topic)
        
        if result["success"]:
            print(f"   ✅ Learned: {result['summary_snippet']}")
            print(f"   ❤️ Reaction: {result['reaction']['dominant_realm']} (Shift: {result['reaction']['emotional_shift']})")
            
            # 2. Reflect / Create
            print("   ✍️  Reflecting (Writing Concept)...")
            
            # We force the 'seed' of the story to be this topic
            concept = writer.brainstorm(seed_idea=topic)
            
            # We override the conflict to be thematic
            concept.conflict = f"A story exploring {lesson['focus']}."
            
            print(f"   📖 Concept Generated: '{concept.title}'")
            print(f"      Theme: {concept.theme}")
            print(f"      Conflict: {concept.conflict}")
            
        else:
            print(f"   ❌ Failed to learn: {result['message']}")
            
        time.sleep(2) # Pause for digestion

    print("\n🎓 Class Dismissed. Elysia is wiser now.")

if __name__ == "__main__":
    study_session()
