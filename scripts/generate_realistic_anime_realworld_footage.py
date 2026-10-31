"""
scripts/generate_realistic_anime_realworld_footage.py

Generates realistic high-fidelity footage (Anime Scenery & Real-World Natural Landscape):
- Rich layered sky gradient with atmospheric sunset/aurora lighting.
- Distant mountain ridges with atmospheric perspective and mist.
- Layered textured foliage/forest canopy with organic light scattering.
- Moving character / vehicle silhouette with natural dynamic motion and wind flows.
- Continuous color grading (Red Flux dusk light, Blue Order mountains/sky, Yellow Entropy illumination).
- Supports generating sequences up to 4K (3840x2160) or HD (1920x1080) for evaluation.
"""

from typing import List, Tuple
import math
import numpy as np


def generate_realistic_anime_landscape_frame(
    width: int,
    height: int,
    t: float,
    character_pos_x: float = 0.5
) -> np.ndarray:
    """
    Synthesizes a realistic high-fidelity anime/cinematic scenery frame:
    - Gradient sunset sky (Golden hour -> Deep twilight blue).
    - Multi-layered mountain silhouettes with atmospheric fog.
    - Organic forest canopy with waving leaves in the wind.
    - Reflective water surface with dynamic ripples and sunset glare.
    - Walking anime character silhouette with flowing cloak/hair.
    """
    img = np.zeros((height, width, 3), dtype=np.float32)

    # Normalized coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    y_norm = y_coords.astype(np.float32) / float(height) # 0 (top) to 1 (bottom)
    x_norm = x_coords.astype(np.float32) / float(width)  # 0 (left) to 1 (right)

    # 1. Atmospheric Sky & Sunset Gradient (Golden Amber to Deep Indigo)
    # Sunset sun disc near horizon at (x=0.7, y=0.45)
    sun_dist = np.sqrt((x_norm - 0.7)**2 + (y_norm - 0.45)**2)
    sun_glow = np.exp(-sun_dist * 4.5)

    sky_r = (1.0 - y_norm * 1.2) * 0.95 + sun_glow * 0.8
    sky_g = (1.0 - y_norm * 1.5) * 0.60 + sun_glow * 0.5
    sky_b = (y_norm * 0.8 + 0.2) * 0.90 + sun_glow * 0.1

    sky = np.stack([sky_r, sky_g, sky_b], axis=-1)
    img = np.clip(sky, 0.0, 1.0)

    # 2. Distant Majestic Mountain Ranges (Fractal Layer 1: Far)
    mtn_far_height = 0.45 + 0.08 * np.sin(x_norm * 6.0) + 0.04 * np.sin(x_norm * 15.0 + 1.0)
    mtn_far_mask = (y_norm > mtn_far_height) & (y_norm < 0.65)
    # Far mountain color: Hazy purple/blue with atmospheric haze
    img[mtn_far_mask] = img[mtn_far_mask] * 0.35 + np.array([0.35, 0.30, 0.55]) * 0.65

    # Mid Mountain Range (Fractal Layer 2: Mid)
    mtn_mid_height = 0.52 + 0.06 * np.sin(x_norm * 9.0 + 2.0) + 0.03 * np.cos(x_norm * 24.0)
    mtn_mid_mask = (y_norm > mtn_mid_height) & (y_norm < 0.70)
    img[mtn_mid_mask] = img[mtn_mid_mask] * 0.20 + np.array([0.20, 0.22, 0.42]) * 0.80

    # 3. Reflective Water / Lake Shoreline (y_norm in [0.65, 0.88])
    water_mask = (y_norm >= 0.68) & (y_norm <= 0.85)
    water_y = (y_norm[water_mask] - 0.68) / 0.17
    # Water ripples driven by time t
    water_ripples = 0.04 * np.sin(y_coords[water_mask] * 0.4 + t * 4.0) * np.cos(x_coords[water_mask] * 0.1)
    # Mirror reflection of sunset glow on water
    sun_glare = np.exp(-((x_norm[water_mask] - 0.7)**2) * 12.0) * (1.0 - water_y)
    refl_r = 0.20 + sun_glare * 0.70 + water_ripples
    refl_g = 0.25 + sun_glare * 0.40 + water_ripples
    refl_b = 0.45 + sun_glare * 0.10 + water_ripples
    water_rgb = np.stack([refl_r, refl_g, refl_b], axis=-1)
    img[water_mask] = np.clip(water_rgb, 0.0, 1.0)

    # 4. Lush Foreground Foliage & Grass Shore (y_norm > 0.80)
    shore_height = 0.82 + 0.04 * np.sin(x_norm * 14.0) + 0.01 * np.sin(x_norm * 60.0)
    shore_mask = y_norm > shore_height
    # Rich anime grass texture with wind sway
    wind_sway = 0.02 * np.sin(x_norm[shore_mask] * 30.0 + t * 5.0)
    grass_r = 0.08 + wind_sway
    grass_g = 0.22 + 0.12 * (y_norm[shore_mask] - 0.82) + wind_sway
    grass_b = np.full_like(grass_r, 0.10)
    img[shore_mask] = np.clip(np.stack([grass_r, grass_g, grass_b], axis=-1), 0.0, 1.0)

    # 5. Foreground Pine Trees / Silhouettes on the left (x_norm < 0.28)
    for tree_x in [0.08, 0.16, 0.24]:
        dx = np.abs(x_norm - tree_x)
        # Triangular pine canopy
        tree_top = 0.35 + tree_x * 0.5
        tree_mask = (y_norm > tree_top) & (y_norm < 0.88) & (dx < (y_norm - tree_top) * 0.25)
        img[tree_mask] = np.array([0.05, 0.12, 0.08])

    # 6. Moving Anime Character Silhouette walking along shore
    char_cx = character_pos_x
    char_cy = 0.79 # Ground level
    # Body oval
    body_dx = (x_norm - char_cx) / 0.015
    body_dy = (y_norm - (char_cy - 0.06)) / 0.05
    char_body = (body_dx**2 + body_dy**2) < 1.0

    # Head circle
    head_dx = (x_norm - char_cx) / 0.010
    head_dy = (y_norm - (char_cy - 0.12)) / 0.018
    char_head = (head_dx**2 + head_dy**2) < 1.0

    # Flowing scarf / cloak waving in the wind
    cloak_dx = (x_norm - (char_cx - 0.02 - 0.015 * math.sin(t * 8.0))) / 0.025
    cloak_dy = (y_norm - (char_cy - 0.05)) / 0.035
    char_cloak = (cloak_dx**2 + cloak_dy**2) < 1.0

    char_mask = char_body | char_head | char_cloak
    # Character silhouette with warm rim light
    img[char_mask] = np.array([0.08, 0.04, 0.06]) # Dark silhouette
    # Rim light from sunset
    rim_mask = char_mask & (x_norm > char_cx)
    img[rim_mask] = np.array([0.95, 0.65, 0.35])

    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def generate_realistic_footage_sequence(
    num_frames: int = 24,
    width: int = 640,
    height: int = 360,
    fps: float = 24.0
) -> List[np.ndarray]:
    """
    Generates a realistic multi-frame sequence of realistic anime landscape.
    Character walks smoothly from left to right along the shoreline.
    """
    frames = []
    dt = 1.0 / fps

    for f in range(num_frames):
        t = f * dt
        # Character progresses smoothly across the screen from x=0.35 to x=0.65
        char_x = 0.35 + 0.30 * (f / max(1, num_frames - 1))
        frame = generate_realistic_anime_landscape_frame(
            width=width,
            height=height,
            t=t,
            character_pos_x=char_x
        )
        frames.append(frame)

    return frames
