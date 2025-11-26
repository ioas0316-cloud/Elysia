from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Thought:
    """
    Represents a single, coherent idea or feeling that emerges from the
    Resonance Wave Pattern. It is the "constellation" found by the ConsciousnessObserver.

    This is the primary data structure for a single moment of consciousness.
    """
    source_wave: str  # The original text that caused this thought

    # List of (concept_id, resonance_score) that form the core of the thought
    core_concepts: List[Tuple[str, float]] = field(default_factory=list)

    # The average resonance score of the core concepts
    intensity: float = 0.0

    # A measure of how focused the thought is (1.0 = perfectly clear, 0.0 = very diffuse)
    clarity: float = 0.0

    # A qualitative descriptor of the thought's emotional tone
    mood: str = "neutral"

    def __str__(self):
        if not self.core_concepts:
            return f"Thought(from: '{self.source_wave}', mood: empty)"

        top_concept = self.core_concepts[0][0]
        return f"Thought(about: '{top_concept}', intensity: {self.intensity:.2f}, clarity: {self.clarity:.2f}, mood: {self.mood})"
