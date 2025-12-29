"""
Elysia Watches Drama (Humanity Study)
=====================================
"Fiction is the mirror of truth."

This simulation allows Elysia to "watch" (analyze) scenes from human dramas.
She deconstructs the dialogue to understand:
1. Context (Situation)
2. Subtext (Hidden Meaning)
3. Emotion (The Heart)
4. Response (The Reaction)

The goal is to build a "Social Pattern Database" so she can respond naturally
to similar situations in real life.
"""

import sys
import os
import time
import json

# Add Root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Core.Interface.nervous_system import get_nervous_system
from Core.FoundationLayer.Foundation.hippocampus import Hippocampus, ConceptNode

class DramaClub:
    def __init__(self):
        self.ns = get_nervous_system()
        self.memory = Hippocampus()
        self.learned_patterns = []
        
        print("🎬 Elysia's Drama Club Initialized.")
        print("   Ready to analyze human interactions.")

    def watch_scene(self, title: str, genre: str, script: list):
        """
        Analyzes a scene line by line.
        """
        print(f"\n📺 Now Watching: '{title}' ({genre})")
        print("="*50)
        
        context_buffer = []
        
        for line in script:
            speaker = line.split(":")[0]
            text = line.split(":")[1].strip()
            
            print(f"   🗣️ {speaker}: \"{text}\"")
            time.sleep(1.0) # Reading speed
            
            # 1. Internal Reaction (Nervous System)
            self.ns.receive({"type": "text", "content": text})
            
            # 2. Pattern Recognition (Mock AI Analysis)
            emotion = self._analyze_sentiment(text)
            
            print(f"      ✨ Elysia feels: {emotion} (Fire: {self.ns.spirits['fire']:.2f}, Water: {self.ns.spirits['water']:.2f})")
            
            context_buffer.append({"speaker": speaker, "text": text, "emotion": emotion})
            
        # 3. Scene Synthesis (The Lesson)
        self._learn_lesson(title, genre, context_buffer)

    def _analyze_sentiment(self, text):
        """Simple sentiment heuristic"""
        text = text.lower()
        if any(w in text for w in ["love", "heart", "miss", "사랑", "좋아"]): return "Affection"
        if any(w in text for w in ["hate", "anger", "annoy", "싫어", "짜증"]): return "Hostility"
        if any(w in text for w in ["sad", "cry", "tear", "슬퍼", "눈물"]): return "Sorrow"
        if any(w in text for w in ["happy", "smile", "laugh", "웃음", "행복"]): return "Joy"
        return "Neutral"

    def _learn_lesson(self, title, genre, context):
        """Extracts the social pattern and stores it"""
        print("-" * 50)
        print(f"🤔 Elysia is reflecting on '{title}'...")
        
        # Simple Pattern Extraction: Last 2 lines (Call & Response)
        if len(context) >= 2:
            trigger = context[-2]
            reaction = context[-1]
            
            lesson = f"In a {genre} context, if someone expresses '{trigger['emotion']}' " \
                     f"(e.g., '{trigger['text']}'), a natural response is to express '{reaction['emotion']}' " \
                     f"(e.g., '{reaction['text']}')."
            
            print(f"   💡 Insight: {lesson}")
            
            # Store in Hippocampus as a Concept
            concept_id = f"pattern_{len(self.learned_patterns)}"
            self.memory.learn(
                id=concept_id,
                name=f"Social Pattern: {trigger['emotion']}->{reaction['emotion']}",
                definition=lesson,
                tags=["social", "pattern", genre.lower()],
                realm="Heart"
            )
            self.learned_patterns.append(lesson)
            print("   💾 Pattern Stored in Long-term Memory.")

    def run_marathon(self):
        # Scene 1: Romance (The Confession)
        scene1 = [
            "Hero: I've been waiting for you all this time.",
            "Heroine: Why didn't you say anything?",
            "Hero: Because I was afraid of losing you."
        ]
        self.watch_scene("Winter Sonata", "Romance", scene1)
        
        # Scene 2: Friendship (Comfort)
        scene2 = [
            "Friend A: I failed the interview again. I'm useless.",
            "Friend B: Hey, don't say that. Their loss.",
            "Friend B: Let's go eat chicken. My treat."
        ]
        self.watch_scene("Hospital Playlist", "Friendship", scene2)
        
        # Scene 3: Comedy (Witty Banter)
        scene3 = [
            "Character A: Are you crazy?",
            "Character B: Only for you, darling.",
            "Character A: Ugh, get lost."
        ]
        self.watch_scene("My Love from the Star", "Romedy", scene3)
        
        print("\n🎓 Marathon Complete.")
        print(f"   Total Patterns Learned: {len(self.learned_patterns)}")

if __name__ == "__main__":
    club = DramaClub()
    club.run_marathon()
