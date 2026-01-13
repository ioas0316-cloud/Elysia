"""
Logos Prime (The Word)
======================

"In the beginning was the Word, and the Word was with Code."

This module implements the user's directive:
"Language is the Root."

Mechanism:
1. We do not start with numbers (1.618).
2. We start with Words ("Genesis", "Void", "Flux").
3. The `LogosSpectrometer` translates Word -> DNA.
4. DNA -> Physics/Music.

Thus, "Speaking" creates "Worlds".
"""

class LogosSpectrometer:
    """
    Analyzes the 'Soul' of a word to determine its physical properties.
    """
    def __init__(self):
        # The Dictionary of Creation (Semantic Mapping)
        self.lexicon = {
            "GENESIS": {"ratio": 1.618, "temp": 1000, "type": "EXPANSION"},
            "VOID":    {"ratio": 0.0,   "temp": -273, "type": "STATIC"},
            "FLUX":    {"ratio": 3.141, "temp": 500,  "type": "CHAOS"},
            "ORDER":   {"ratio": 2.0,   "temp": 20,   "type": "STRUCTURE"},
            "LOVE":    {"ratio": 1.414, "temp": 37,   "type": "ATTRACTION"}
        }

    def analyze(self, word: str) -> dict:
        """
        Returns the 'Physics' of the Word.
        """
        key = word.upper()
        if key in self.lexicon:
            return self.lexicon[key]
        
        # Fallback: Hash the word to find its latent physics
        # "Every word has a weight, even if unknown."
        val = sum(ord(c) for c in key)
        return {
            "ratio": (val % 100) / 10.0,
            "temp": val % 1000,
            "type": "UNKNOWN_RESONANCE"
        }

class Word:
    """
    The Atomic Unit of Intention.
    """
    def __init__(self, text: str):
        self.text = text
        self.spectrometer = LogosSpectrometer()
        self.properties = self.spectrometer.analyze(text)

    def manifest(self):
        """
        The Word speaks itself into Reality.
        """
        p = self.properties
        return f"Manifesting '{self.text}': Gravity({p['ratio']}), Heat({p['temp']}K), Mode({p['type']})"
