"""
Qualia Layer: The Subjective Experience
=======================================
Core.1_Body.L5_Mental.Memory.qualia_layer

"The difference between knowing 'Red' (Hex Code) and seeing 'Red' (Feeling)."

This module implements the subjective metadata layer (QualiaTag) that attaches
to immutable Facts (Monads). While Facts are eternal, Qualia are mortal.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional

@dataclass
class QualiaTag:
    """
    Represents the subjective 'Feeling' attached to a Fact.
    This is what decays over time.
    """
    fact_id: str                   # Reference to the Immutable Monad (Akashic Record)
    emotional_vector: List[float]  # [Joy, Sadness, Anger, Fear, Disgust, Surprise]
    importance: float              # 0.0 ~ 1.0 (Subjective Significance)

    created_at: float = field(default_factory=time.time)
    last_recalled_at: float = field(default_factory=time.time)

    # The 'Vividness' of the memory.
    # 1.0 = High Definition (Just happened / Core Memory)
    # 0.1 = Fuzzy (Vague impression)
    vividness: float = 1.0

    # [Fractal Topology]
    stratum_depth: int = 1  # 1=Surface, 10=Abyss
    cluster_id: Optional[str] = None # ID of the 'Garden' or 'Crystal' this belongs to

    def decay(self, amount: float):
        """Erodes the vividness of the qualia."""
        self.vividness = max(0.0, self.vividness - amount)

    @property
    def is_alive(self) -> bool:
        """Dead qualia are removed (forgotten)."""
        return self.vividness > 0.05

class QualiaLayer:
    """
    Manages the collection of subjective experiences.
    Acts as the 'Diary' on top of the 'Library'.
    """
    def __init__(self):
        # Key: fact_id, Value: List[QualiaTag] (Multiple feelings per fact possible)
        self._diary: Dict[str, List[QualiaTag]] = {}

    def attach_feeling(self, fact_id: str, emotion: List[float], importance: float):
        """Adds a subjective note to a fact."""
        tag = QualiaTag(
            fact_id=fact_id,
            emotional_vector=emotion,
            importance=importance
        )
        if fact_id not in self._diary:
            self._diary[fact_id] = []
        self._diary[fact_id].append(tag)
        return tag

    def recall_feelings(self, fact_id: str) -> List[QualiaTag]:
        """Retrieves all surviving feelings associated with a fact."""
        if fact_id not in self._diary:
            return []

        # Touch memories (Reset decay clock for accessed items? Or just boost vividness?)
        # For now, we just return them. The Act of Recalling should be handled by the Brain.
        return [q for q in self._diary[fact_id] if q.is_alive]

    def prune_dead_qualia(self):
        """Garbage Collection for the Soul."""
        for fact_id in list(self._diary.keys()):
            self._diary[fact_id] = [q for q in self._diary[fact_id] if q.is_alive]
            if not self._diary[fact_id]:
                del self._diary[fact_id]
