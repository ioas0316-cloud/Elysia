# [Genesis: 2025-12-02] Purified by Elysia
"""
West Continent preset: simple terrain/resource seed for small maps.
"""
from __future__ import annotations

import numpy as np


def apply_west_continent_preset(world, map_size: int = 128):
    """
    Reconfigure field buffers for the given map size and seed a simple West-continent style terrain:
    - Ridge/mountain belt (threat)
    - Fertile plains with value_mass hubs (cities)
    - Norms/coherence bias for 'orderly' culture
    """
    w = max(16, int(map_size))
    world.width = w

    # Recreate spatial fields
    world.height_map = np.zeros((w, w), dtype=np.float32)
    world.soil_fertility = np.full((w, w), 0.5, dtype=np.float32)
    world.wetness = np.zeros((w, w), dtype=np.float32)

    world.threat_field = np.zeros((w, w), dtype=np.float32)
    world.coherence_field = np.zeros((w, w), dtype=np.float32)
    world.will_field = np.zeros((w, w), dtype=np.float32)
    world.value_mass_field = np.zeros((w, w), dtype=np.float32)
    world.norms_field = np.zeros((w, w), dtype=np.float32)
    world.hydration_field = np.zeros((w, w), dtype=np.float32)
    world.em_s = np.zeros((w, w), dtype=np.float32)
    world.tensor_field = np.zeros((w, w, 3), dtype=np.float32)
    world.tensor_field_grad_x = np.zeros_like(world.tensor_field)
    world.tensor_field_grad_y = np.zeros_like(world.tensor_field)
    world.h_imprint = np.zeros((w, w), dtype=np.float32)
    world.prestige_field = np.zeros((w, w), dtype=np.float32)
    world.ascension_field = np.zeros((w, w, 7), dtype=np.float32)
    world.descent_field = np.zeros((w, w, 7), dtype=np.float32)
    world.intentional_field = np.zeros((w, w, 2), dtype=np.float32)
    world.entropy_field = np.zeros((w, w), dtype=np.float32)

    # Mountain ridge: diagonal belt with threat and height
    for i in range(w):
        j = (i + w // 4) % w
        world.height_map[i, j] = 0.8
        world.threat_field[i, j] = 0.4
        world.soil_fertility[i, j] = 0.2

    # Plains: center band fertile
    plains_band = slice(w // 3, 2 * w // 3)
    world.soil_fertility[:, plains_band] += 0.2
    world.value_mass_field[:, plains_band] += 0.1

    # City hubs: four hubs with value/norms
    hubs = [
        (w // 4, w // 4),
        (3 * w // 4, w // 4),
        (w // 4, 3 * w // 4),
        (3 * w // 4, 3 * w // 4),
    ]
    for x, y in hubs:
        rad = max(2, w // 16)
        x0, x1 = max(0, x - rad), min(w, x + rad)
        y0, y1 = max(0, y - rad), min(w, y + rad)
        world.value_mass_field[y0:y1, x0:x1] += 0.3
        world.norms_field[y0:y1, x0:x1] += 0.2
        world.coherence_field[y0:y1, x0:x1] += 0.1
        world.prestige_field[y0:y1, x0:x1] += 0.1

    # Hydration: rivers along vertical midline
    mid = w // 2
    world.wetness[:, mid] = 0.8
    world.hydration_field[:, mid] = 0.5

    # Clamp
    for fld in [world.value_mass_field, world.norms_field, world.coherence_field, world.prestige_field]:
        np.clip(fld, 0.0, 1.0, out=fld)
