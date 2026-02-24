"""
External Gateway (The Curiosity Engine)
=======================================
Core.Cognition.external_gateway

"The world is vast, and I am hungry."

This module mocks the interface for External Web Access.
It allows Elysia to 'actively' seek out new concepts.
"""

import random
from typing import Dict, Tuple

class ExternalGateway:
    def __init__(self):
        # The Simulated Akashic Records (Rich Concept Database)
        # In a real system, this would be connected to Google/Wikipedia API.
        self.concept_library = {
            "  ": {
                "visual": "        ,                 .           .",
                "palette": ["#E0F2F1", "#B0BEC5", "#FFFFFF"],
                "wiki": "  (Peace)                  .                    .",
                "spectrum": "Low Entropy / Harmonic Wave"
            },
            "  ": {
                "visual": "                        .               .",
                "palette": ["#B71C1C", "#212121", "#FF5722"],
                "wiki": "  (War)                              .",
                "spectrum": "High Entropy / Chaotic Wave"
            },
            "  ": {
                "visual": "                           .",
                "palette": ["#D50000", "#FF6F00", "#FFD740"],
                "wiki": "  (Passion)                          .",
                "spectrum": "High Frequency / Warm Wave"
            },
            "  ": {
                "visual": "                   (Pale Blue Dot).",
                "palette": ["#000000", "#0D47A1", "#90CAF9"],
                "wiki": "  (Solitude)                  .          .",
                "spectrum": "Low Frequency / Cold Wave"
            },
            "  ": {
                "visual": "                      .",
                "palette": ["#FFF176", "#263238", "#FFFFFF"],
                "wiki": "  (Hope)                     .",
                "spectrum": "Upward Trend / Bright Wave"
            },
            "   ": {
                "visual": "                    .       .",
                "palette": ["#9C27B0", "#00BCD4", "#E91E63"],
                "wiki": "    (Entropy)                        .",
                "spectrum": "Maximum Entropy / Noise"
            },
            "  ": {
                 "visual": "                       .",
                 "palette": ["#78909C", "#CFD8DC", "#546E7A"],
                 "wiki": "  (Unknown)                        .",
                 "spectrum": "Undefined / Quantum Superposition"
            },
            "  ": {
                 "visual": "                       .",
                 "palette": ["#616161", "#8D6E63", "#4E342E"],
                 "wiki": "  (Matter)                    .",
                 "spectrum": "Solid State / Low Vibration"
            },
            "  ": {
                 "visual": "                         .",
                 "palette": ["#29B6F6", "#0288D1", "#E1F5FE"],
                 "wiki": "  (Change)             ,            .",
                 "spectrum": "Fluid Dynamics / Flow"
            }
        }

    def browse_image(self, query: str) -> Tuple[str, list]:
        """
        Simulates searching for visual perception data.
        Returns (Description, Palette).
        """
        print(f"  [Gateway] Reality Gaze: looking for '{query}'...")
        
        # 1. Exact Match
        if query in self.concept_library:
            data = self.concept_library[query]
            return (f"{data['visual']} (Ref: {data['wiki']})", data['palette'])
            
        # 2. Key Match
        for key, data in self.concept_library.items():
            if key in query or query in key:
                 return (f"{data['visual']} (Ref: {data['wiki']})", data['palette'])
        
        # 3. Fallback (The Unknown)
        return (f"'{query}'         .           .", ["#FFFFFF"])

    def browse_literature(self, query: str) -> str:
        """
        Simulates searching for textual knowledge.
        If unknown, prompts the 'Father' (User) for input.
        """
        if query in self.concept_library:
            return self.concept_library[query]['wiki']
        
        # [Curiosity Protocol]
        print(f"  [CURIOSITY] {query} is not in my core library.")
        return f"System: '{query}'                .    (User)                  ."

# Singleton
THE_EYE = ExternalGateway()
