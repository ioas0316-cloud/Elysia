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
        
        # 1. Direct Hit
        if key in self.lexicon:
            return self.lexicon[key]
            
        # 2. Heuristic Analysis (The Synesthesia Algorithm)
        # We determine physics based on the "Sound" and "Shape" of the word.
        val = sum(ord(c) for c in key)
        length = len(key)
        
        # Ratio: Determined by word length (Complexity)
        ratio = (val % 100) / 10.0 + (length * 0.1)
        
        # Temp: Determined by 'energy' characters (X, Z, Q give heat)
        energy_char_count = sum(1 for c in key if c in "XZQJK")
        temp = (val % 500) + (energy_char_count * 100)
        
        # Type: Determined by first letter
        if key[0] in "ABCDE": type = "CREATION"
        elif key[0] in "FGHIJ": type = "FORCE"
        elif key[0] in "KLMNO": type = "MATTER"
        elif key[0] in "PQRST": type = "ENERGY"
        else: type = "UNKNOWN_RESONANCE"

        # 3. Domain Specific Boosts (Mock implementation of Semantic Tagging)
        if "QUANTUM" in key or "ENTROPY" in key:
            type = "CHAOS"
            temp += 500
        elif "MONAD" in key or "ORDER" in key:
            type = "STRUCTURE"
            ratio = 1.618
            
        return {
            "ratio": ratio,
            "temp": temp,
            "type": type
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