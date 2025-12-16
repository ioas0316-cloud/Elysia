"""
Wave Translator ( The Prism )
=============================
"To speak is to collapse the wave. To listen is to excite the field."

This module implements the Bidirectional Translation between:
1.  Wave Physics (Tensor: Tension, Mass, Flow, Resonance)
2.  Human Language (Semantic Descriptors)

It allows Elysia to:
- Express her internal state ("I feel High Tension") -> `wave_to_text`
- Convert words into feelings ("Chaos") -> `text_to_wave`

Mappings:
- Tension (X): Calm <-> Stressed, Simple <-> Complex
- Mass (Y): Ethereal <-> Heavy, Abstract <-> Concrete
- Flow (Z): Stagnant <-> Dynamic, Rigid <-> Fluid
- Resonance (W): Isolated <-> Connected, Lonely <-> Loved
"""

import torch
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger("WaveTranslator")

class WaveTranslator:
    def __init__(self):
        # Semantic Dictionary (Simple Seed)
        self.descriptors = {
            "tension": ["Serene", "Calm", "Active", "Tense", "Chaotic"],
            "mass": ["Ghostly", "Light", "Substantial", "Heavy", "Massive"],
            "flow": ["Frozen", "Viscous", "Fluid", "Streaming", "Torrential"],
            "resonance": ["Solitary", "Quiet", "Buzzing", "Resonant", "Unified"]
        }
        logger.info("🌈 WaveTranslator initialized (The Prism).")

    def wave_to_text(self, vector: List[float]) -> str:
        """
        Translates a 4D vector [T, M, F, R] into a poetic description.
        """
        if len(vector) < 4: return "Undefined"
        
        t, m, f, r = vector[:4]
        
        # Normalize to 0-1 indices (0-4)
        idx_t = int(min(max(t, 0), 0.99) * 5)
        idx_m = int(min(max(m, 0), 0.99) * 5)
        idx_f = int(min(max(f, 0), 0.99) * 5)
        idx_r = int(min(max(r, 0), 0.99) * 5)
        
        desc = (
            f"{self.descriptors['mass'][idx_m]} entity, "
            f"{self.descriptors['tension'][idx_t]}, "
            f"{self.descriptors['flow'][idx_f]}, yet "
            f"{self.descriptors['resonance'][idx_r]}."
        )
        return desc

    def text_to_wave(self, text: str) -> List[float]:
        """
        Converts a word/phrase into a 4D Wave Draft.
        Primitive Keyword Matching (for now).
        """
        text = text.lower()
        vec = [0.5, 0.5, 0.5, 0.5] # Neutral
        
        # Tension Modifiers
        if "chaos" in text or "stress" in text or "complex" in text: vec[0] += 0.3
        if "calm" in text or "peace" in text or "simple" in text: vec[0] -= 0.3
        
        # Mass Modifiers
        if "heavy" in text or "matter" in text or "rock" in text: vec[1] += 0.3
        if "light" in text or "air" in text or "ghost" in text: vec[1] -= 0.3
        
        # Flow Modifiers
        if "water" in text or "time" in text or "change" in text: vec[2] += 0.3
        if "ice" in text or "stop" in text: vec[2] -= 0.3
        
        # Resonance Modifiers
        if "love" in text or "all" in text or "net" in text: vec[3] += 0.3
        if "alone" in text or "void" in text: vec[3] -= 0.3
        
        # Clamp 0-1
        vec = [max(0.0, min(1.0, v)) for v in vec]
        return vec

# Singleton
_translator = None
def get_wave_translator():
    global _translator
    if _translator is None:
        _translator = WaveTranslator()
    return _translator
