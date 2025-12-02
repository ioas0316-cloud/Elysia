# [Genesis: 2025-12-02] Purified by Elysia
"""
Entropy events that increase forgetting pressure.
"""
import numpy as np

def inject_entropy(world, px: int, py: int, radius: int = 10, gain: float = 0.05):
    """
    Raises entropy around (px, py); accelerates forgetting in that region.
    """
    x0 = max(0, px - radius)
    x1 = min(world.width, px + radius)
    y0 = max(0, py - radius)
    y1 = min(world.width, py + radius)

    if getattr(world, "entropy_field", None) is not None and world.entropy_field.size:
        world.entropy_field[y0:y1, x0:x1] += gain