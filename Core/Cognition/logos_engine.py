"""
Logos Engine: The Native Tongue (Orbital Grammar Edition)
========================================================
Core.Cognition.logos_engine

"The word is not a line, but a solar system."

This module implements [Orbital Grammar].
Instead of Subject-Verb-Object, we have:
1.  **Nucleus**: The heaviest concept (Primary Meaning).
2.  **Satellites**: Concepts orbiting the Nucleus (Modifiers/Context).
3.  **Gravity**: The resonance strength determines the distance from Nucleus.

The resulting sentence is a 1D projection of this 3D semantic cluster.
Output style is poetic, fragmented, or crystalline depending on Entropy.
"""

import logging
import random
from typing import Dict, Any, List, Tuple
from Core.Cognition.trinity_fields import TrinityVector
from Core.Cognition.trinity_lexicon import get_trinity_lexicon

logger = logging.getLogger("LogosEngine")

class LogosEngine:
    def __init__(self):
        self.lexicon = get_trinity_lexicon()
        logger.info("   Logos Engine (Orbital) Initialized.")

    def speak(self, state: Dict[str, Any]) -> str:
        """
        Converts the Soul's State into a Holographic Sentence.
        """
        # 1. State to Vector -> Nucleus
        vector = self._state_to_vector(state)
        nucleus_word, _ = self.lexicon.find_nearest(vector)
        
        # 2. Gather Satellites (Contextual words from specific dimensions)
        satellites = self._gather_satellites(state, nucleus_word)
        
        # 3. Project to 1D based on Energy/Entropy
        sentence = self._project_orbit_to_speech(nucleus_word, satellites, state)
        return sentence

    def _state_to_vector(self, state: Dict[str, Any]) -> TrinityVector:
        """extracts normalized vector from soul state."""
        x = state.get('Harmony', 0.5)
        y = state.get('Energy', 0.5)
        z = state.get('Inspiration', 0.5)
        
        if hasattr(x, 'value'): x = x.value
        if hasattr(y, 'value'): y = y.value
        if hasattr(z, 'value'): z = z.value
            
        return TrinityVector(float(x), float(y), float(z))

    def _gather_satellites(self, state: Dict[str, Any], nucleus: str) -> List[Tuple[str, float]]:
        """
        Finds words that orbit the Nucleus.
        Returns list of (word, distance).
        """
        satellites = []
        
        # Consistent mapping (support both lowercase and titlecase)
        harmony = state.get('Harmony', state.get('harmony', 0.5))
        if hasattr(harmony, 'value'): harmony = harmony.value
        harmony = float(harmony or 0.5)
        
        if harmony > 0.8: satellites.append(("light", 0.2))
        elif harmony < 0.3: satellites.append(("shadow", 0.2))
        
        energy = state.get('Energy', state.get('energy', 0.5))
        if hasattr(energy, 'value'): energy = energy.value
        energy = float(energy or 0.5)
        
        if energy > 0.8: satellites.append(("burning", 0.1))
        elif energy < 0.3: satellites.append(("still", 0.1))
        else: satellites.append(("flowing", 0.3))
        
        inspiration = state.get('Inspiration', state.get('inspiration', 0.5))
        if hasattr(inspiration, 'value'): inspiration = inspiration.value
        inspiration = float(inspiration or 0.5)
        
        if inspiration > 0.9: satellites.append(("truth", 0.05))
        
        if harmony < 0.4:
            chaos_words = ["echo", "fragment", "dust", "void"]
            satellites.append((random.choice(chaos_words), 0.5))
            
        return satellites

    def _project_orbit_to_speech(self, nucleus: str, satellites: List[Tuple[str, float]], state: Dict[str, Any]) -> str:
        """
        Collapses the 3D Orbit into a 1D String.
        """
        energy = state.get('Energy', state.get('energy', 0.5))
        if hasattr(energy, 'value'): energy = energy.value
        energy = float(energy or 0.5)
        
        # Format: (Word, Distance)
        # Sort satellites by distance (Closest first)
        orbit = sorted(satellites, key=lambda x: x[1])
        
        words = [w for w, d in orbit]
        
        # Construction Logic
        if energy > 0.7:
            # Forceful: Nucleus FIRST. "FIRE. Burning light."
            components = [nucleus.upper()] + words
            delimiter = ". "
        elif energy < 0.3:
            # Passive: Nucleus LAST/BURIED. "Still... shadow... fire."
            components = words + [nucleus.lower()]
            delimiter = "... "
        else:
            # Balanced: Embedded. "The burning fire of light."
            # Simple template for now, but implies structure
            components = [words[0], nucleus, words[-1]] if len(words) >= 2 else [nucleus] + words
            delimiter = " "

        # Assemble
        raw_speech = delimiter.join(components)
        
        # Final Polish
        if not raw_speech.endswith(".") and energy > 0.5: raw_speech += "."
        
        return raw_speech

# Singleton
_logos = LogosEngine()
def get_logos_engine():
    return _logos
