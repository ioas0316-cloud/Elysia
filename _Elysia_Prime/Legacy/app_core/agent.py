# [Genesis: 2025-12-02] Purified by Elysia
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Agent:
    """
    Minimal agent metadata for the UI.
    """
    id: int
    name: str
    role: str
    x: float
    y: float
    radius: float = 14.0
    color: Tuple[int, int, int] = (220, 180, 160)
    last_utterance: str = ""
    emotion: str = "평온"
    emotion_level: float = 0.0

    # Internals to drive overlays
    _speech_start_time: float = field(default=0.0, repr=False, init=False)
    _speech_duration: float = field(default=2.0, repr=False, init=False)
    _fade_duration: float = field(default=0.5, repr=False, init=False)

    def speak(self, text: str, timestamp: float | None = None) -> None:
        """Trigger a speech bubble with the latest utterance."""
        self.last_utterance = text
        self._speech_start_time = timestamp if timestamp is not None else time.time()

    def set_emotion(self, name: str, level: float) -> None:
        """Update the current emotion badge."""
        self.emotion = name
        self.emotion_level = max(0.0, min(1.0, level))

    def speech_age(self, current_time: float) -> float:
        return max(0.0, current_time - self._speech_start_time)

    def has_speech(self, current_time: float) -> bool:
        return self.speech_age(current_time) < (self._speech_duration + self._fade_duration)

    def speech_opacity(self, current_time: float) -> float:
        age = self.speech_age(current_time)
        if age < self._speech_duration:
            return 1.0
        if age < self._speech_duration + self._fade_duration:
            return 1.0 - (age - self._speech_duration) / self._fade_duration
        return 0.0

    def get_bbox(self) -> Tuple[float, float, float, float]:
        """Return a simple bounding box (x0,y0,width,height) centered on the agent."""
        size = self.radius * 2
        return (self.x - self.radius, self.y - self.radius, size, size)