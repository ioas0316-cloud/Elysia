"""
Spectrum mapping helpers.

- value_to_hue: map scalar (0..1 or 0..100) to hue degrees (0..360).
- efp_to_color: map Energy/Force/Persistence triple to RGB tuple (0..255).
"""
from __future__ import annotations

from typing import Tuple


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def value_to_hue(value: float, value_range: Tuple[float, float] = (0.0, 100.0)) -> float:
    """
    Maps a scalar value to hue degrees.
    Default: 0 -> 0 deg (red), 100 -> 360 deg (~red), mid -> green/blue range.
    """
    v_min, v_max = value_range
    if v_max == v_min:
        return 0.0
    t = (value - v_min) / (v_max - v_min)
    t = _clamp01(t)
    return t * 360.0


def efp_to_color(energy: float, force: float, persistence: float) -> Tuple[int, int, int]:
    """
    Simple mapping of E/F/P (0..1) to RGB (0..255).
    - Energy -> Red
    - Force -> Green
    - Persistence -> Blue
    """
    r = int(_clamp01(energy) * 255)
    g = int(_clamp01(force) * 255)
    b = int(_clamp01(persistence) * 255)
    return r, g, b
