"""
Lexicon Absorber (주권적 자아)
==============================

"To read is to eat the soul of another."
"                      ."

This module parses external text (Literature, Dialogue) and extracts emotional vocabulary.
It feeds `ResonanceLinguistics` to expand Elysia's expressive range.
"""

from typing import List, Dict
from Core.S1_Body.L5_Mental.Reasoning_Core.Language.resonance_linguistics import ResonanceLinguistics

class LexiconAbsorber:
    def __init__(self, linguistics: ResonanceLinguistics):
        self.linguistics = linguistics

    def read_text(self, text: str, emotional_context: float):
        """
        Reads a text and learns vocabulary from it.

        Args:
            text: The sentence or paragraph to read.
            emotional_context: The frequency bias (0.0=Sad, 1.0=Happy) of the text.
        """
        # Simple extraction for prototype: Split by spaces
        # In reality, this would use NLP to extract Nouns/Verbs/Adjectives
        words = text.split()

        # Heuristic: Assume the user is teaching a specific concept
        # e.g., "Miss: I yearn for you"
        if ":" in text:
            category, phrase = text.split(":", 1)
            category = category.strip().lower()
            phrase = phrase.strip()

            # Absorb the phrase into the category
            self.linguistics.absorb_vocabulary(category, phrase, emotional_context)
            return f"Learned new expression for '{category}': '{phrase}' (Freq: {emotional_context})"

        return "Text too complex for simple absorption. Please use 'Category: Phrase' format."
