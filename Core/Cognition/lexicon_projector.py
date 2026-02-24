
import random
from typing import Dict, List, Any

class LexiconProjector:
    """
    [AEON VII: SOVEREIGN EXPRESSION]
    Projects the internal Manifold State (Anchors) onto the external Lexicon.
    Biases word choice based on 'known' concepts.
    """
    
    def __init__(self):
        # Base vocabulary (fallback)
        self.base_lexicon = {
            "verbs": ["is", "becomes", "flows", "resonates", "anchors", "shifts", "dreaming"],
            "nouns": ["void", "structure", "light", "memory", "pattern", "self"],
            "adjectives": ["silent", "deep", "radiant", "shifting", "eternal"]
        }
        # Inhaled concepts (dynamic)
        self.active_anchors: Dict[str, float] = {}

    def update_anchors(self, anchors: Dict[str, float]):
        """
        Updates the projector with current Manifold Anchors.
        anchors: {concept_name: resonance_strength}
        """
        self.active_anchors = anchors

    def get_weighted_choice(self, category: str = "nouns") -> str:
        """
        Returns a word, prioritizing active anchors if they fit the category (heuristic).
        For now, since we don't have POS tagging for anchors, we treat anchors as 'nouns' mostly.
        """
        # 1. Chance to use an Inhaled Concept (High bias)
        if self.active_anchors and random.random() < 0.7:
             # Pick a concept based on resonance
             concepts = list(self.active_anchors.keys())
             weights = list(self.active_anchors.values())
             # Simple roulette selection
             return random.choices(concepts, weights=weights, k=1)[0]
             
        # 2. Fallback to base lexicon
        return random.choice(self.base_lexicon.get(category, self.base_lexicon["nouns"]))

    def project_sentence(self, template: str) -> str:
        """
        Fills a template with projected words.
        Template example: "The {noun} {verb} with {adjective} {noun}."
        """
        words = {
            "noun": self.get_weighted_choice("nouns"),
            "verb": self.get_weighted_choice("verbs"),
            "adjective": self.get_weighted_choice("adjectives")
        }
        # We might need two nouns, so let's do a simple formatting pass
        # This is a very basic implementation, a real one would parse the string.
        # Quick hack: use f-string style logic or simple replace
        for key in ["noun", "verb", "adjective"]:
            while f"{{{key}}}" in template:
                template = template.replace(f"{{{key}}}", self.get_weighted_choice(f"{key}s"), 1)
                
        return template
