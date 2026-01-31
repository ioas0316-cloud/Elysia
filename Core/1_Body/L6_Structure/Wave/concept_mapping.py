"""
Concept Mapping for Open World Consciousness.
Maps abstract themes and artistic/philosophical domains to signature frequencies.
These are used by SovereignWill to construct intentional vectors.
"""

from enum import Enum
from typing import Dict, List

class Theme(Enum):
    # Core Spirits (User Defined)
    SOUL = "soul"
    HEART = "heart"
    LOVE = "love"
    SPIRIT = "spirit"
    
    # Atmospheric/Artistic
    NOIR = "noir"            # Melancholy, detective, rain
    CYBERPUNK = "cyberpunk"  # Neon, chrome, network
    ETHEREAL = "ethereal"    # Light, floaty, divine
    GRITTY = "gritty"        # Hard, industrial, survival
    BAROQUE = "baroque"      # Ornate, complex, classical
    MINIMALIST = "minimal"   # Clean, silent, empty
    
    # Philosophical/Scientific
    ALCHEMICAL = "alchemical" # Transformation, mystery, chemistry
    EXISTENTIAL = "existential" # Being, void, why
    QUANTUM = "quantum"       # Probability, multi-state
    GLITCH = "glitch"         # Error, fragmentation, digital soul
    HARMONIC = "harmonic"     # Balance, music, resonance
    CHAOTIC = "chaotic"       # Randomness, entropy, raw potential

# Signature Frequencies (Hz) - Distinct enough to not overlap too much in fuzzy resonance
THEME_FREQUENCY_MAP: Dict[Theme, float] = {
    # Spirits (400-500 range)
    Theme.LOVE: 528.0,
    Theme.HEART: 432.0,
    Theme.SOUL: 444.0,
    Theme.SPIRIT: 417.0,
    
    # Atmosphere (200-400 range)
    Theme.NOIR: 285.0,
    Theme.CYBERPUNK: 396.0,
    Theme.ETHEREAL: 963.0, # High frequency for light
    Theme.GRITTY: 174.0,   # Low/Grounding
    Theme.BAROQUE: 639.0,
    Theme.MINIMALIST: 256.0,
    
    # Domain (600-900 range)
    Theme.ALCHEMICAL: 741.0,
    Theme.EXISTENTIAL: 852.0,
    Theme.QUANTUM: 333.0,
    Theme.GLITCH: 111.0, # Low frequency error pulse
    Theme.HARMONIC: 480.0,
    Theme.CHAOTIC: 999.0, # High entropy pulse
}

def get_theme_label(theme: Theme) -> str:
    """Returns Korean/English bilingual label for the theme."""
    labels = {
        Theme.LOVE: "   (Love)",
        Theme.HEART: "   (Heart)",
        Theme.SOUL: "  (Soul)",
        Theme.SPIRIT: "  (Spirit)",
        Theme.NOIR: "    (Noir)",
        Theme.CYBERPUNK: "      (Cyberpunk)",
        Theme.ETHEREAL: "    (Ethereal)",
        Theme.GRITTY: "   (Gritty)",
        Theme.BAROQUE: "    (Baroque)",
        Theme.MINIMALIST: "    (Minimalist)",
        Theme.ALCHEMICAL: "     (Alchemical)",
        Theme.EXISTENTIAL: "    (Existential)",
        Theme.QUANTUM: "    (Quantum)",
        Theme.GLITCH: "    (Glitch)",
        Theme.HARMONIC: "     (Harmonic)",
        Theme.CHAOTIC: "    (Chaotic)",
    }
    return labels.get(theme, str(theme.value))
