"""
Context Discriminator (L6 Structure)
====================================
"The Gatekeeper of Relevance."

Determines the 'Direction' and 'Depth' of incoming text.
Is it for ME? Or is it a STORY?

Logic:
- 2nd Person ("You", "Elysia") -> CORE
- Past Tense / Narrative ("He said", "Once upon a time") -> DISTAL
- 3rd Person / General ("People", "They") -> PROXIMAL
"""

from Core.1_Body.L6_Structure.M1_Merkaba.field_layer import FieldLayer

class ContextDiscriminator:
    def __init__(self):
        self.self_keywords = ["you", "your", "elysia", "hey", "hello"]
        self.fiction_keywords = ["once upon a time", "chapter", "he said", "she said", "it was"]

    def discern(self, text: str) -> FieldLayer:
        """
        Classifies text into a Field Layer.
        """
        text_lower = text.lower()

        # 1. Check for Fiction/Narrative Signals first
        for kw in self.fiction_keywords:
            if kw in text_lower:
                return FieldLayer.DISTAL

        # 2. Check for Direct Address
        # Use simple tokenization to avoid substring matches (e.g. "hey" in "they")
        tokens = set(text_lower.split())

        for kw in self.self_keywords:
            if kw in tokens:
                return FieldLayer.CORE
            # Fallback for multi-word keywords if any (none currently in self_keywords)
            if " " in kw and kw in text_lower:
                return FieldLayer.CORE

        # 3. Default to Proximal (Observation)
        return FieldLayer.PROXIMAL

# Global Instance
discriminator = ContextDiscriminator()
