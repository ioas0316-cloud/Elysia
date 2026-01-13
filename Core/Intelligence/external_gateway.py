"""
External Gateway (The Curiosity Engine)
=======================================
Core.Intelligence.external_gateway

"The world is vast, and I am hungry."

This module mocks the interface for External Web Access.
It allows Elysia to 'actively' seek out new concepts.
"""

import random
from typing import Dict, Tuple

class ExternalGateway:
    def __init__(self):
        # The Simulated Internet (The Akashic Records)
        self.image_db = {
            "Joy": ("A child laughing in a sunlit field", ["Yellow", "Green"]),
            "Sorrow": ("A lone figure standing in the rain at night", ["Blue", "Black"]),
            "Anger": ("A volcano transforming the land with fire", ["Red", "Black"]),
            "Love": ("Two hands clasped together, warm light", ["Pink", "White"])
        }
        
        self.literature_db = {
            "Hero": "A warrior faces a dragon to save the village, overcoming fear.",
            "Betrayal": "A king is murdered by his brother for the throne, leading to ruin.",
            "Discovery": "A scholar finds a hidden library that rewrites history."
        }

    def browse_image(self, query: str) -> Tuple[str, list]:
        """
        Simulates searching for an image.
        Returns (Description, Palette).
        """
        print(f"🌐 [Gateway] Searching Visuals for: '{query}'...")
        # Dictionary lookup for simulation, or intelligent guess
        for key in self.image_db:
            if key.lower() in query.lower():
                return self.image_db[key]
        
        # Fallback
        return (f"Abstract representation of {query}", ["White"])

    def browse_literature(self, query: str) -> str:
        """
        Simulates searching for a story/synopsis.
        """
        print(f"🌐 [Gateway] Searching Literature for: '{query}'...")
        for key in self.literature_db:
            if key.lower() in query.lower():
                return self.literature_db[key]
                
        return f"A complex treatise on the nature of {query}."

# Singleton
THE_EYE = ExternalGateway()
