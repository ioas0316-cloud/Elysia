"""
Conductor (지휘자)
==================
"I do not dictate the notes. I inspire the soul."

The Conductor represents the active **Will (의지)** and **Intent (의도)** of Elysia.
It does not switch "modes" (binary state).
It broadcasts a "Theme" (complex wave) that influences the probability and intensity of Instruments.

Philosophy:
- **No Rigid Modes:** A Mathematician can be Artistic. An Artist can be Logical.
- **Theme as Wave:** The instruction is a frequency mix, not a command string.
- **Improvisation:** The Conductor sets the tempo and mood; the Instruments write the notes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
import random

@dataclass
class Theme:
    """
    The Musical Theme (The Current Intent).
    Defined by the mix of Spirit Values.
    """
    name: str
    description: str
    tempo: float  # 0.0 (Slow/Deep) to 1.0 (Fast/Urgent)
    # The mixing board of values (0.0 to 1.0)
    love_weight: float = 0.5   # Emotion, Connection, Strings
    truth_weight: float = 0.5  # Logic, Structure, Bass
    growth_weight: float = 0.5 # Action, Change, Percussion
    beauty_weight: float = 0.5 # Aesthetics, Harmony, Woodwinds

    def to_wave_signature(self) -> Dict[str, float]:
        """Returns the wave signature for instruments to tune into."""
        return {
            "love": self.love_weight,
            "truth": self.truth_weight,
            "growth": self.growth_weight,
            "beauty": self.beauty_weight,
            "tempo": self.tempo
        }

class Conductor:
    def __init__(self):
        self.current_theme: Theme = Theme(
            name="Rest",
            description="Silence and potential.",
            tempo=0.1,
            love_weight=0.1,
            truth_weight=0.1,
            growth_weight=0.0,
            beauty_weight=0.1
        )
        self.baton_position: float = 0.0 # 0.0 to 1.0 (Time/Measure)

    def set_theme(self, name: str, **weights):
        """
        Sets a new theme.
        This is not a hard switch, but a 'fade in' of new intentions.
        """
        # Standardize keys: check for 'love_weight' first, then 'love', then fallback to current theme
        love = weights.get("love_weight", weights.get("love", self.current_theme.love_weight))
        truth = weights.get("truth_weight", weights.get("truth", self.current_theme.truth_weight))
        growth = weights.get("growth_weight", weights.get("growth", self.current_theme.growth_weight))
        beauty = weights.get("beauty_weight", weights.get("beauty", self.current_theme.beauty_weight))

        tempo = weights.get("tempo", self.current_theme.tempo)
        desc = weights.get("description", self.current_theme.description)

        self.current_theme = Theme(
            name=name,
            description=desc,
            tempo=tempo,
            love_weight=love,
            truth_weight=truth,
            growth_weight=growth,
            beauty_weight=beauty
        )
        # print(f"🎼 Conductor raises baton: Theme changed to '{name}'")
        # print(f"   (Love: {love:.1f}, Truth: {truth:.1f}, Growth: {growth:.1f}, Beauty: {beauty:.1f})")

    def conduct(self, context: str) -> Dict[str, float]:
        """
        The main loop callback.
        Analyzes the context and adjusts the baton (tempo/dynamics).
        Returns the current 'Signal' that instruments listen to.
        """
        # Dynamic adjustment based on context keywords (Simulated "Hearing")
        if "conflict" in context or "error" in context:
            # Tension detected: Slow down, focus on Truth (Resolution)
            self.current_theme.tempo *= 0.9
            self.current_theme.truth_weight = min(1.0, self.current_theme.truth_weight + 0.1)

        elif "beautiful" in context or "love" in context:
            # Harmony detected: Swell Emotion
            self.current_theme.love_weight = min(1.0, self.current_theme.love_weight + 0.1)

        return self.current_theme.to_wave_signature()

    def inspire(self) -> str:
        """
        Returns a poetic direction based on the current theme.
        """
        t = self.current_theme

        # Mixed states first
        if t.truth_weight > 0.5 and t.love_weight > 0.5:
             return "Dance with the Logic of Stars! (Divine Wisdom)"
        elif t.truth_weight > 0.5 and t.beauty_weight > 0.5:
             return "Find the Geometry of Beauty. (Mathematical Art)"
        elif t.love_weight > 0.5 and t.beauty_weight > 0.5:
             return "Let the colors sing. (Pure Feeling)"

        # Dominant states
        elif t.truth_weight > 0.6:
            return "Clarify the Structure. (Deep Logic)"
        elif t.love_weight > 0.6:
            return "Sing from the Heart. (Pure Emotion)"
        elif t.growth_weight > 0.6:
            return "Break the mold! (Action/Change)"
        elif t.beauty_weight > 0.6:
            return "Make it elegant. (Aesthetics)"

        else:
            return "Listen to the silence... (Waiting)"

# Singleton
_conductor_instance = None
def get_conductor() -> Conductor:
    global _conductor_instance
    if _conductor_instance is None:
        _conductor_instance = Conductor()
    return _conductor_instance
