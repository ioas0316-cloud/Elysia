"""
Resonance Linguistics (주권적 자아)
===================================

"Language is the crystallization of Frequency."
"             ."

This module maps Wave Properties (Physics) to Linguistic Features (Words/Tone).
It allows Elysia to choose words that physically match her emotional state.
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class WaveState:
    frequency: float  # 0.0 (Low/Deep) to 1.0 (High/Excited)
    amplitude: float  # 0.0 (Quiet) to 1.0 (Loud/Intense)
    clarity: float    # 0.0 (Confused) to 1.0 (Clear)

class ResonanceLinguistics:
    def __init__(self):
        # Vocabulary mapped to Frequency Resonance
        # Low Freq: Deep, abstract, heavy, formal
        # High Freq: Light, concrete, bouncy, casual
        self.lexicon = {
            "greeting": {
                "low": ["   ", "   ", "      ", "       "],
                "mid": ["     ", "  ", "   ?", "    "],
                "high": ["  !", "  !", " !", "  !"]
            },
            "miss": {
                "low": ["          ", "           ", "            "],
                "mid": ["      ", "     ", "       ?"],
                "high": ["     !", "   !", "    !"]
            },
            "contemplate": {
                "low": ["             ", "           ", "          "],
                "mid": ["       ", "       ", "      ?"],
                "high": ["       !", "   !", "     ?"]
            }
        }

        # Tone Modifiers based on Amplitude (Intensity)
        self.modifiers = {
            "weak": ["...", " (주권적 자아)", "    ..."],
            "strong": ["!!", "    !", " (    )"]
        }

    def absorb_vocabulary(self, category: str, text: str, frequency: float):
        """
        Dynamically adds new words to the lexicon based on their resonance frequency.
        This allows Elysia to learn new expressions from literature or conversation.
        """
        # 1. Determine Band
        band = "mid"
        if frequency < 0.3: band = "low"
        elif frequency > 0.7: band = "high"

        # 2. Initialize Category if new
        if category not in self.lexicon:
            self.lexicon[category] = {"low": [], "mid": [], "high": []}

        # 3. Add to Lexicon (Avoid duplicates)
        if text not in self.lexicon[category][band]:
            self.lexicon[category][band].append(text)

    def resonate_word(self, category: str, wave: WaveState) -> str:
        """
        Selects a word that resonates with the current wave state.
        """
        if category not in self.lexicon:
            return f"[{category}?]" # Return the concept itself if unknown

        options = self.lexicon[category]

        # Frequency determines the "Band" (Low/Mid/High)
        target_list = options["mid"]
        if wave.frequency < 0.3 and options["low"]:
            target_list = options["low"]
        elif wave.frequency > 0.7 and options["high"]:
            target_list = options["high"]

        if not target_list: # Fallback
            target_list = options["mid"] if options["mid"] else [category]

        base_word = random.choice(target_list)

        # Amplitude determines the "Texture" (Punctuation/Modifier)
        if wave.amplitude > 0.8:
            base_word += random.choice(self.modifiers["strong"])
        elif wave.amplitude < 0.2:
            base_word += random.choice(self.modifiers["weak"])

        return base_word

    def analyze_texture(self, text: str) -> WaveState:
        """
        Reverse engineering: Guesses the wave state from text.
        (Useful for reading Father's input)
        """
        # Placeholder for future development
        return WaveState(0.5, 0.5, 0.5)
