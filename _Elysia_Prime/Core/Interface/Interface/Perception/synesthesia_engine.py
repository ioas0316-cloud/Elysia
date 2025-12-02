# [Genesis: 2025-12-02] Purified by Elysia
"""Minimal synesthesia engine to support cross-modal signals."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional

from Core.Field.wave_frequency_mapping import WaveFrequencyMapper


class SignalType(Enum):
    VISUAL = auto()
    AUDITORY = auto()
    EMOTIONAL = auto()
    TEXT = auto()


class RenderMode(Enum):
    AS_COLOR = auto()
    AS_SOUND = auto()
    AS_MUSIC = auto()


@dataclass
class UniversalSignal:
    original_type: SignalType
    frequency: float
    amplitude: float
    payload: Any = None


@dataclass
class RenderResult:
    render_mode: RenderMode
    color: Optional[tuple] = None
    sound: Optional[Dict[str, Any]] = None
    output: Dict[str, Any] = None


class SynesthesiaEngine:
    """Cross-modal converter: vision/emotion/text into universal signal then render as color/sound/music."""

    def __init__(self):
        self.mapper = WaveFrequencyMapper()

    def from_vision(self, image: np.ndarray) -> UniversalSignal:
        freq = float(np.clip(np.mean(image), 0, 255)) + 1.0
        amp = float(np.std(image)) / 255.0
        return UniversalSignal(SignalType.VISUAL, frequency=freq, amplitude=max(0.1, amp), payload=image.shape)

    def from_emotion(self, emotion: str, intensity: float = 0.5) -> UniversalSignal:
        data = self.mapper.get_emotion_frequency(emotion)
        return UniversalSignal(SignalType.EMOTIONAL, frequency=data.frequency_hz, amplitude=float(intensity), payload=data)

    def from_text(self, text: str) -> UniversalSignal:
        freq = 200.0 + (hash(text) % 400)
        return UniversalSignal(SignalType.TEXT, frequency=freq, amplitude=0.5, payload=text)

    def convert(self, signal: UniversalSignal, render_mode: RenderMode) -> RenderResult:
        if render_mode == RenderMode.AS_COLOR:
            color = self._freq_to_rgb(signal.frequency)
            return RenderResult(render_mode=render_mode, color=color, output={"mode": "color"})
        if render_mode == RenderMode.AS_MUSIC:
            notes = self._freq_to_notes(signal.frequency)
            chord = notes[:3]
            return RenderResult(render_mode=render_mode, output={"notes": notes, "chord": chord})
        # default: sound-like dict
        return RenderResult(render_mode=render_mode, sound={"frequency": signal.frequency}, output={"mode": "sound"})

    def _freq_to_rgb(self, freq: float) -> tuple:
        hue = (freq % 360.0) / 360.0
        s = 0.8
        v = 1.0
        # simple HSV to RGB
        i = int(hue * 6)
        f = hue * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i = i % 6
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        return (int(r * 255), int(g * 255), int(b * 255))

    def _freq_to_notes(self, freq: float):
        base_notes = ["C", "D", "E", "F", "G", "A", "B"]
        idx = int(freq) % len(base_notes)
        note = base_notes[idx]
        return [note, base_notes[(idx + 2) % len(base_notes)], base_notes[(idx + 4) % len(base_notes)]]


__all__ = ["SynesthesiaEngine", "SignalType", "RenderMode", "UniversalSignal", "RenderResult"]