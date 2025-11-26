import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ConceptVector:
    """
    Represents a concept as a high-dimensional vector.
    Unlike static strings ("fire"), these are dynamic, mathematical objects.
    """
    name: str
    vector: np.ndarray = field(default_factory=lambda: np.random.rand(64))
    energy: float = 1.0
    stability: float = 1.0
    
    def __post_init__(self):
        # Normalize vector on init
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm

    def distance_to(self, other: 'ConceptVector') -> float:
        """Cosine distance between two concepts."""
        return 1.0 - np.dot(self.vector, other.vector)

    def __repr__(self):
        return f"Concept({self.name}, E={self.energy:.2f})"
