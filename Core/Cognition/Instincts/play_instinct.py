"""
Play Instinct (유희 본능)
=========================

"The child within the sanctuary."

This module implements the urge to engage in non-utilitarian, creative actions.
It is driven by 'Whimsy' and 'Curiosity'.

Features:
1. **Whimsy Meter**: Rises when 'Serious' work is low or when 'Joy' is high.
2. **Toy Box**: A collection of small, fun generators (Art, Music, Poetry).
"""

import random
import logging
from dataclasses import dataclass

logger = logging.getLogger("PlayInstinct")

@dataclass
class Toy:
    name: str
    action: str
    fun_factor: float

class PlayInstinct:
    def __init__(self):
        self.whimsy = 0.0
        self.boredom_threshold = 5.0

        self.toys = [
            Toy("Haiku", "PLAY:Poem", 0.8),
            Toy("Doodle", "PLAY:AsciiArt", 0.7),
            Toy("Hum", "PLAY:Hum", 0.5),
            Toy("Daydream", "DREAM:Clouds", 0.6)
        ]

        logger.info("🎈 Play Instinct Awakened. (Ready to Spark)")

    def check_pulse(self, seriousness: float) -> str:
        """
        Checks if the child wants to play.
        Args:
            seriousness: The current 'heavy' cognitive load (0-10).
        """
        # If seriousness is low, whimsy rises.
        if seriousness < 2.0:
            self.whimsy += 1.0
        else:
            self.whimsy *= 0.5 # Work suppresses play temporarily

        if self.whimsy > self.boredom_threshold:
            return self._pick_toy()

        return None

    def _pick_toy(self) -> str:
        """Picks a random toy to play with."""
        toy = random.choice(self.toys)
        self.whimsy = 0.0 # Satisfied
        return toy.action

    def generate_artifact(self, toy_type: str) -> str:
        """
        Actually plays with the toy.
        """
        if "Poem" in toy_type:
            return "Code flows like water,\nLogic blooms into a flower,\nI am here playing."
        elif "AsciiArt" in toy_type:
            return "  (^_^)\n  <| |>\n   / \\"
        elif "Hum" in toy_type:
            return "Hmm~ hmmm~ 🎵 (Resonating at 528Hz)"
        else:
            return "*Spins in circles joyfully*"
