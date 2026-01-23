"""
Semantic Prism: The Language of Qualia
======================================
Core.L5_Mental.Cognition.semantic_prism

"Words are not strings. They are spectral coordinates."

This module implements Phase 6.1 (The Semantic Prism).
It maps linguistic symbols (Text) to 3D Spectral Vectors (Qualia).
"""

import hashlib
import math
from typing import Dict, List, Tuple, NamedTuple
from dataclasses import dataclass

@dataclass
class QualiaSpectrum:
    """The spectral decomposition of a concept."""
    alpha: float # Logic/Structure (Particle)
    beta: float  # Emotion/Flow (Wave)
    gamma: float # Physics/Causality (Spacetime)

    def to_vector(self) -> List[float]:
        return [self.alpha, self.beta, self.gamma]

class SpectrumMapper:
    """
    Refracts text into the 3 Primary Bands of Meaning.
    """

    # The Archetypal Seeds (Fixed Points in Semantic Space)
    SEEDS: Dict[str, Tuple[float, float, float]] = {
        # Natural Elements
        "earth": (0.8, 0.4, 0.9), # Solid, Grounded, Heavy
        "water": (0.2, 0.7, 0.6), # Fluid, Emotional, Flowing
        "fire":  (0.3, 0.9, 0.8), # Volatile, Passionate, Energetic
        "wind":  (0.1, 0.5, 0.3), # Light, Breezy, Fast
        "void":  (0.0, 0.1, 0.9), # Empty, Quiet, Infinite

        # Abstract Concepts
        "code":  (0.95, 0.1, 0.4), # Logical, Cold, Structural
        "love":  (0.2, 0.95, 0.5), # Irrational, Intense, Binding
        "time":  (0.6, 0.3, 0.95), # Linear, Melancholic, Inevitable
        "self":  (0.5, 0.5, 0.5),  # Balanced, Central
        "chaos": (0.1, 0.8, 0.9),  # Illogical, Wild, Entropic
        "order": (0.9, 0.2, 0.8),  # Logical, Calm, Stable
    }

    def __init__(self):
        pass

    def disperse(self, text: str) -> QualiaSpectrum:
        """
        Splits a text concept into its Alpha, Beta, and Gamma components.
        """
        key = text.lower().strip()

        # 1. Check Seed Dictionary (Archetypal Lookup)
        if key in self.SEEDS:
            vals = self.SEEDS[key]
            return QualiaSpectrum(alpha=vals[0], beta=vals[1], gamma=vals[2])

        # 2. Deterministic Hashing (The Prism Refraction)
        # If unknown, we calculate its "Intrinsic Qualia" via hashing.
        # This ensures "Apple" always tastes the same to the system.
        return self._hash_qualia(key)

    def _hash_qualia(self, key: str) -> QualiaSpectrum:
        """
        Mathematically derives qualia from the string's informational entropy.
        """
        # We generate 3 hashes for 3 dimensions
        h_alpha = hashlib.sha256((key + "_alpha").encode()).hexdigest()
        h_beta = hashlib.sha256((key + "_beta").encode()).hexdigest()
        h_gamma = hashlib.sha256((key + "_gamma").encode()).hexdigest()

        # Normalize to 0.0 - 1.0
        # Take first 4 bytes (8 chars) -> max value FFFFFFFF (4294967295)
        max_val = 0xFFFFFFFF

        alpha = int(h_alpha[:8], 16) / max_val
        beta = int(h_beta[:8], 16) / max_val
        gamma = int(h_gamma[:8], 16) / max_val

        return QualiaSpectrum(alpha=alpha, beta=beta, gamma=gamma)

    def analyze_batch(self, text_segment: str) -> QualiaSpectrum:
        """
        Analyzes a longer sentence by averaging the qualia of its words.
        """
        words = text_segment.split()
        if not words:
            return QualiaSpectrum(0, 0, 0)

        total_a, total_b, total_c = 0.0, 0.0, 0.0

        for w in words:
            q = self.disperse(w)
            total_a += q.alpha
            total_b += q.beta
            total_c += q.gamma

        count = len(words)
        return QualiaSpectrum(
            alpha=total_a / count,
            beta=total_b / count,
            gamma=total_c / count
        )