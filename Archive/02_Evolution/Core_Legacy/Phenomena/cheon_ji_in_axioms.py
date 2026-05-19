"""
Cheon-Ji-In Axioms (천지인 공리)
===============================
Core.Phenomena.cheon_ji_in_axioms

"The vowels follow the Heaven, Earth, and Man.
 The consonants follow the Five Elements."
"모음은 천지인을 따르고, 자음은 오행을 따른다."

This module defines the metaphysical 'A Priori' knowledge that Elysia possesses
before any experience. It allows her to deduce what sound *should* match a 
given physical phenomenon.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum

class Element(Enum):
    WOOD = "WOOD"   # Spring, Start, Explosive (Velar)
    FIRE = "FIRE"   # Summer, Bloom, Flowing/Lingual (Lingual)
    EARTH = "EARTH" # Late Summer, Center, Containment (Labial)
    METAL = "METAL" # Autumn, Harvest, Cutting/Dental (Dental)
    WATER = "WATER" # Winter, Storage, Deep/Guttural (Guttural)

class CheonJiIn(Enum):
    HEAVEN = "HEAVEN" # (•) Yang, Bright, Upward, Outward
    EARTH = "EARTH"   # (ㅡ) Yin, Dark, Flat, Inward
    MAN = "MAN"       # (ㅣ) Neutral, Standing, Vertical

@dataclass
class MetaphysicalProperty:
    element: Element
    polarity: float # 1.0 (Yang) to -1.0 (Yin)
    interaction: str # "Expansion", "Contraction", "Flow", "Impact"

class AxiomaticEngine:
    """
    The engine that maps physical qualities to metaphysical axioms.
    """
    def __init__(self):
        # 1. Consonant Axioms (Five Elements)
        # Based on Hunminjeongeum Haerye
        self.consonant_map = {
            Element.WOOD: ["ㄱ", "ㅋ", "ㄲ"], # Velar (Tooth root blocks throat)
            Element.FIRE: ["ㄴ", "ㄷ", "ㅌ", "ㄸ"], # Lingual (Tongue touches upper palate)
            Element.EARTH: ["ㅁ", "ㅂ", "ㅍ", "ㅃ"], # Labial (Mouth shape)
            Element.METAL: ["ㅅ", "ㅈ", "ㅊ", "ㅆ", "ㅉ"], # Dental (Teeth shape)
            Element.WATER: ["ㅇ", "ㅎ"], # Guttural (Throat open)
        }

        # 2. Vowel Axioms (Cheon-Ji-In)
        # Derived from Heaven(•), Earth(ㅡ), Man(ㅣ) interaction
        self.vowel_map = {
            "YANG_MAX": ["ㅏ", "ㅑ", "ㅗ", "ㅛ", "ㅐ", "ㅒ"], # Heaven dominant
            "YIN_MAX": ["ㅓ", "ㅕ", "ㅜ", "ㅠ", "ㅔ", "ㅖ"], # Earth dominant
            "NEUTRAL": ["ㅣ", "ㅡ", "ㅢ"], # Man dominant or balanced
        }

    def deduce_consonant(self, roughness: float, temperature: float, flow: float) -> str:
        """
        Deduces the consonant based on elemental physics.
        
        Logic:
        - High Roughness (Impact) -> Wood (K) or Metal (S/J)
        - High Temperature (Active) -> Fire (N/D)
        - High Flow (Fluid) -> Water (H/O) or Fire
        - High Stability (Containment) -> Earth (M/B)
        """
        # Determine Element
        element = Element.EARTH # Default
        
        if roughness > 0.8:
            element = Element.WOOD if temperature > 0.5 else Element.METAL
        elif flow > 0.8:
            element = Element.WATER
        elif temperature > 0.7:
            element = Element.FIRE
        
        # Select specific Jamo based on intensity
        candidates = self.consonant_map[element]
        idx = min(len(candidates)-1, int(roughness * len(candidates)))
        return candidates[idx]

    def deduce_vowel(self, openness: float, brightness: float) -> str:
        """
        Deduces the vowel based on Cheon-Ji-In.
        
        Logic:
        - High Brightness -> Yang (Heaven) -> 'ㅏ', 'ㅗ'
        - Low Brightness -> Yin (Earth) -> 'ㅓ', 'ㅜ'
        - Neutral -> 'ㅣ', 'ㅡ'
        - High Openness -> Vertical (Standing Man/Outward)
        - Low Openness -> Horizontal (Flat Earth)
        """
        if brightness > 0.6: # Yang
            # Outward (Open) vs Upward (Closed but high)
            return "ㅏ" if openness > 0.5 else "ㅗ"
        elif brightness < 0.4: # Yin
            # Inward (Open) vs Downward (Closed)
            return "ㅓ" if openness > 0.5 else "ㅜ"
        else: # Neutral
            return "ㅣ" if openness > 0.5 else "ㅡ"
