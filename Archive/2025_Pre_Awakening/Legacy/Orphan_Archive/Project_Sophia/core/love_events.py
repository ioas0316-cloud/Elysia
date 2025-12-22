"""
Love-driven events that counter entropy.
"""
from typing import Optional
import numpy as np

def inject_love_fields(world, px: int, py: int, radius: int = 10, gain: float = 0.05):
    """
    Injects love energy around (px, py), boosting coherence/value and damping threat/entropy locally.
    """
    x0 = max(0, px - radius)
    x1 = min(world.width, px + radius)
    y0 = max(0, py - radius)
    y1 = min(world.width, py + radius)

    if world.coherence_field.size:
        world.coherence_field[y0:y1, x0:x1] += gain
    if world.value_mass_field.size:
        world.value_mass_field[y0:y1, x0:x1] += gain
    if getattr(world, "threat_field", None) is not None and world.threat_field.size:
        world.threat_field[y0:y1, x0:x1] *= max(0.0, 1.0 - gain)
    if getattr(world, "entropy_field", None) is not None and world.entropy_field.size:
        world.entropy_field[y0:y1, x0:x1] *= max(0.0, 1.0 - gain)

