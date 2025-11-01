from dataclasses import dataclass
from typing import Optional, List
from copy import deepcopy

@dataclass
class EmotionalState:
    valence: float  # -1 (매우 부정) ~ 1 (매우 긍정)
    arousal: float  # 0 (평온) ~ 1 (흥분)
    dominance: float  # -1 (복종) ~ 1 (지배)
    primary_emotion: str
    secondary_emotions: List[str]

    def copy(self):
        """Creates a deep copy of this emotional state."""
        return deepcopy(self)
