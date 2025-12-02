# [Genesis: 2025-12-02] Purified by Elysia
from __future__ import annotations

import pygame
import pygame.freetype as ft
from typing import List

from .layers import LAYERS, toggle
from .fonts import get_font


def handle_layer_keys(e: pygame.event.Event) -> None:
    if e.type != pygame.KEYDOWN:
        return
    key = e.key
    if key == pygame.K_a:  # Agents
        toggle("agents")
    elif key == pygame.K_s:  # Structures
        toggle("structures")
    elif key == pygame.K_f:  # Flora
        toggle("flora")
    elif key == pygame.K_w:  # Will Field
        toggle("will_field")
    elif key == pygame.K_z:  # Fauna (use 'z' to avoid clash)
        toggle("fauna")


def _enabled_tags() -> List[str]:
    tags = []
    if LAYERS.get("agents"): tags.append("A")
    if LAYERS.get("structures"): tags.append("S")
    if LAYERS.get("flora"): tags.append("F")
    if LAYERS.get("fauna"): tags.append("a")
    if LAYERS.get("will_field"): tags.append("W")
    return tags


def draw_layer_hud(screen: pygame.Surface) -> None:
    font = get_font(16)
    text = f"Layers: {','.join(_enabled_tags()) or '-'}"
    surf, _ = font.render(text, fgcolor=(235, 235, 245))
    rect = surf.get_rect()
    rect.topright = (screen.get_width() - 12, 12)
    panel = pygame.Surface((rect.width + 12, rect.height + 8), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 130))
    panel.blit(surf, (6, 4))
    screen.blit(panel, (rect.right - (rect.width + 12), rect.top))


def draw_layer_panel(screen: pygame.Surface) -> None:
    font = get_font(14)
    lines = [
        "[A]gents  [S]truct  [F]lora  Faun[a]  [W]ill",
    ]
    surf_lines = [font.render(s, fgcolor=(220, 220, 230))[0] for s in lines]
    w = max(s.get_width() for s in surf_lines) + 12
    h = sum(s.get_height() for s in surf_lines) + 12
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 140))
    y = 6
    for s in surf_lines:
        panel.blit(s, (6, y))
        y += s.get_height()
    screen.blit(panel, (screen.get_width() - w - 12, screen.get_height() - h - 12))
