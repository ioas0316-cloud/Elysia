# [Genesis: 2025-12-02] Purified by Elysia
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class EmotionalState:
    valence: float  # -1 (매우 부정) ~ 1 (매우 긍정)
    arousal: float  # 0 (평온) ~ 1 (흥분)
    dominance: float  # -1 (복종) ~ 1 (지배)
    primary_emotion: str
    secondary_emotions: List[str]