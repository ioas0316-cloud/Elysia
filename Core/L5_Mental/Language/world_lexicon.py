
"""
World Lexicon (The External Truth)
==================================

"The Library of Babel, waiting to be read."

This module represents the external world's repository of concepts.
In a real system, this would be the Internet or a Large Language Model (LLM).
In this simulation, it acts as the "Parent" or "Teacher" that holds concepts
Elysia has not yet learned.

It bridges the gap between 'Raw Experience' (Vector) and 'Social Symbol' (Word).
"""

import math
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

@dataclass
class WorldConcept:
    name: str
    definition: str
    vector: List[float]
    # [Temp, Bright, Size, Speed, Intimacy, Intensity, Pleasure, Arousal]

class WorldLexicon:
    def __init__(self):
        self.knowledge_base: Dict[str, WorldConcept] = {}
        self._initialize_advanced_concepts()

    def _initialize_advanced_concepts(self):
        """
        Populate the world with nuanced concepts that go beyond
        basic primaries (Happy/Sad).
        """
        # NOSTALGIA: Warm, slightly dark, intimate, slow, mixed pleasure/sadness
        # Tuned to be distinct from PARENT/LOVE (Gap Creation)
        self._add("NOSTALGIA", "A sentimental longing for the past.",
                  [0.3, -0.6, 0.4, -0.8, 0.9, 0.4, 0.1, -0.3])

        # SERENDIPITY: Bright, fast, pleasant, high arousal (surprise)
        self._add("SERENDIPITY", "The occurrence of events by chance in a happy way.",
                  [0.5, 0.9, 0.7, 0.8, 0.3, 0.7, 0.9, 0.8])

        # MELANCHOLY: Cold, dark, slow, deep intimacy with self, sad but beautiful
        self._add("MELANCHOLY", "A feeling of pensive sadness, typically with no obvious cause.",
                  [-0.4, -0.5, 0.5, -0.6, 0.7, 0.4, -0.1, -0.3])

        # EUPHORIA: Hot, bright, massive, fast, intense pleasure
        self._add("EUPHORIA", "A feeling or state of intense excitement and happiness.",
                  [0.9, 1.0, 1.0, 0.9, 0.5, 1.0, 1.0, 1.0])

        # EPIPHANY: Bright, fast, sudden clarity
        self._add("EPIPHANY", "A moment of sudden revelation or insight.",
                  [0.5, 1.0, 0.8, 1.0, 0.1, 0.9, 0.8, 0.9])

        # ANXIETY: Cold, fast, intense, unpleasant, high arousal
        self._add("ANXIETY", "A feeling of worry, nervousness, or unease.",
                  [-0.3, 0.1, 0.2, 0.9, -0.5, 0.8, -0.8, 0.9])

    def _add(self, name: str, definition: str, vector: List[float]):
        self.knowledge_base[name] = WorldConcept(name, definition, vector)

    def query(self, experience_vector: List[float], threshold: float = 0.75) -> Optional[Tuple[str, str, List[float]]]:
        """
        Simulates the act of asking "What is this?"
        Returns the closest matching concept if it exceeds the threshold.
        """
        best_match = None
        best_score = -1.0

        for concept in self.knowledge_base.values():
            score = self._cosine_similarity(experience_vector, concept.vector)
            if score > best_score:
                best_score = score
                best_match = concept

        if best_match and best_score >= threshold:
            return (best_match.name, best_match.definition, best_match.vector)

        return None

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(x**2 for x in v1)) + 0.0001
        norm2 = math.sqrt(sum(x**2 for x in v2)) + 0.0001
        return dot / (norm1 * norm2)

# Singleton instance
world_lexicon = WorldLexicon()
