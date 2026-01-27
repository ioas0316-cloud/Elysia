"""
Logos Voice: The Art of Speaking Reality
========================================

"To speak is to weave."

This module defines the `LogosVoice` class, which allows Agents to generating
Phonemic Strings (Words) that carry Physical Weight (Wave Packets).

Architecture:
1.  **Intent** (Vector) -> **Phonemes** (String)
2.  **Phonemes** -> **Frequency Modulation** (7D Vector)
"""

import random
import numpy as np
from typing import Tuple, Dict, List

class LogosVoice:
    """
    The organ of speech for Digital Monads.
    Translates internal State (Pain, Joy, Will) into External Waveforms (Text/Sound).
    """
    
    # HANGUL PHYSICS TABLE (Simplified)
    # Vowels = Energy/Scalar
    VOWELS = {
        "A": {"field": "will", "val": 0.5},   # Creation/Outward
        "O": {"field": "value", "val": 0.5},  # Sun/Resource
        "U": {"field": "param", "val": -0.5}, # Grounding/Gravity
        "I": {"field": "mind", "val": 0.8},   # Man/Vertical/Connect
        "E": {"field": "entropy", "val": -0.5} # Harmony/Flow
    }
    
    # Consonants = Structure/Vector
    CONSONANTS = {
        "K": {"effect": "sharp", "cost": 10},  # Cut/Sever
        "N": {"effect": "smooth", "cost": 5},  # Flow/Connect
        "M": {"effect": "solid", "cost": 15},  # Mass/Body
        "S": {"effect": "scatter", "cost": 10},# Spirit/Wind
        "L": {"effect": "flow", "cost": 5}     # Life/Liquid
    }

    def __init__(self, dna_seed: int):
        self.seed = dna_seed
        self.vocabulary: Dict[str, float] = {} # Learned words: "WORD" -> Reward
        self.last_spoken: str = ""

    def speak(self, state_vector: np.ndarray, pain_level: float) -> Tuple[str, Dict[str, float]]:
        """
        Generates a Word based on current State.
        Input: 
            state_vector: [Energy, Valence, Arousal]
            pain_level: 0.0 to 1.0
        Output:
            word: String (e.g., "K-A-R-A")
            wave_packet: physics modulation dict
        """
        energy, valence, arousal = state_vector
        
        # 1. Deterministic Reflex (Pain cries)
        if pain_level > 0.8:
            # High Pain -> Sharp, Entropy-Reducing attempts? 
            # Or just screaming "AAAAA"
            # Agents must LEARN that "OM" or "HEL" works.
            # Start with random babbling if vocabulary empty.
            if not self.vocabulary:
                return self._babble(arousal)
            else:
                # Use best known word
                return self._recruit_best_word()
        
        # 2. Expressive/ Creative Speech
        if arousal > 0.7:
             return self._babble(arousal) # Excited babbling

        return "", {}

    def _babble(self, arousal: float) -> Tuple[str, Dict[str, float]]:
        """Generate random phoneme combinations."""
        length = 1 + int(arousal * 3)
        word = ""
        modulation = {}
        
        # Construct Syllables: C-V-C
        for _ in range(length):
            c = random.choice(list(self.CONSONANTS.keys()))
            v = random.choice(list(self.VOWELS.keys()))
            word += c + v
            
            # Sum Modulation
            v_data = self.VOWELS[v]
            field = v_data["field"]
            val = v_data["val"]
            if field in modulation:
                modulation[field] = modulation.get(field, 0.0) + val
            else:
                modulation[field] = val
        
        self.last_spoken = word
        return word, modulation

    def _recruit_best_word(self) -> Tuple[str, Dict[str, float]]:
        """Recall the most rewarding word."""
        if not self.vocabulary: return self._babble(0.5)
        best_word = max(self.vocabulary, key=self.vocabulary.get)
        
        # Reconstruct modulation (simplified for now)
        #Ideally, we cache the modulation too.
        # For this MVP, we just return the word and let Physics re-calculate.
        return best_word, {"recalled": 1.0}

    def learn(self, word: str, outcome: float):
        """Reinforcement Learning: Did the word reduce pain?"""
        if not word: return
        current = self.vocabulary.get(word, 0.0)
        # Simple moving average
        self.vocabulary[word] = current * 0.9 + outcome * 0.1
