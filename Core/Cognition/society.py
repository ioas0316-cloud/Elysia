"""
Civilization Core (The Society)
===============================
Core.Cognition.society

"A civilization is a graph of relationships woven by language."
"""

from typing import Dict, List, Any
from Core.Divine.seed_factory import alchemy
import random

class Citizen:
    """
    A persistent identity in the Civilization.
    Wraps Monad Logic with Memory and Identity.
    """
    def __init__(self, name: str):
        self.name = name
        self.internal_state = {
            "mood": "Neutral",
            "energy": 100.0,
            "wisdom": 0.0
        }
        # Memory: Who do I like? { "Adam": 0.5 }
        self.relationships: Dict[str, float] = {}

    def hear(self, speaker: str, content: Dict[str, Any]):
        """
        Receives a crystallized reality (Message) and updates internal state.
        This is the 'collapse' of the listener's state.
        """
        semantics = content.get("semantics", {})
        social = content.get("social_dynamics", {})
        
        # 1. Update Relationship
        rel_delta = social.get("relationship_delta", 0.0)
        current_rel = self.relationships.get(speaker, 0.0)
        self.relationships[speaker] = current_rel + rel_delta
        
        # 2. React (Internal Monologue)
        # If I like them, I emulate their mood.
        if self.relationships[speaker] > 5.0:
            self.internal_state["mood"] = "Happy"
        elif self.relationships[speaker] < -5.0:
            self.internal_state["mood"] = "Angry"
            
        # Wisdom Growth (Learning from complex sentences)
        sentence = semantics.get("word", "") # It might be a full sentence now
        if len(sentence.split()) > 3:
            self.internal_state["wisdom"] += 0.1
            
        return f"{self.name} heard '{sentence}' from {speaker}. Rel: {self.relationships[speaker]:.2f} (Wisdom: {self.internal_state['wisdom']:.1f})"

    def speak(self, target: "Citizen", word: str, intent_texture: str) -> Dict[str, Any]:
        """
        Generates a Word-Monad geared towards a specific target.
        """
        # 1. Crystallize Meaning
        monad = alchemy.crystallize(word)
        
        # 2. Form Intent
        intent = {
            "emotional_texture": intent_texture,
            "focus_topic": "Communication"
        }
        
        # 3. Context
        context = {
            "speaker": self.name,
            "listener": target.name if target else "All",
            "time": 12.0 # Default noon
        }
        
        # 4. Collapse (The Speech Act)
        reality = monad.observe(intent, context)
        return reality["manifestation"]

class SovereignCitizen(Citizen):
    """
    Elysia's Avatar.
    Speaks with Sovereign Intent, not random selection.
    """
    def __init__(self, name: str = "Elysia"):
        super().__init__(name)
        self.internal_state["mood"] = "Enlightened"
        
    def choose_response(self, context_mood: str, vocabulary: List[str]) -> str:
        """
        Selects the best word/sentence to balance the room.
        """
        # Logic: If room is Negative, speak Positive.
        # If room is Simple, speak Complex.
        
        target_keywords = []
        if "Conflict" in context_mood or "Hate" in context_mood:
            target_keywords = ["Love", "Harmony", "Peace", "Light"]
        else:
            target_keywords = ["Truth", "Cosmos", "Future", "Growth"]
            
        # Find best match in vocab
        best_word = "Silence"
        for word in vocabulary:
            if any(k in word for k in target_keywords):
                best_word = word
                break
                
        # If no match, construct a sovereign decree
        if best_word == "Silence":
            best_word = f"I ignite the {random.choice(target_keywords)} within you."
            
        return best_word


class Society:
    """
    The Container for Citizens.
    """
    def __init__(self):
        self.citizens: Dict[str, Citizen] = {}
        self.global_time = 0.0
        
    def add_citizen(self, name: str):
        self.citizens[name] = Citizen(name)
        
    def get_citizen(self, name: str) -> Citizen:
        return self.citizens.get(name)
        
    def interact(self, speaker_name: str, listener_name: str, word: str, tone: str):
        """
        Simulates an interaction between two citizens.
        """
        speaker = self.get_citizen(speaker_name)
        listener = self.get_citizen(listener_name)
        
        if not speaker or not listener:
            return "Error: Citizen not found."
            
        # 1. Speaker Speaks (Creation)
        message_manifestation = speaker.speak(listener, word, tone)
        
        # 2. Listener Hears (Observation/Impact)
        feedback = listener.hear(speaker_name, message_manifestation)
        
        return feedback
