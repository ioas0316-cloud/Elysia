from __future__ import annotations

import pygame
from pygame import Surface
from pygame import Rect


def _wrap_text(text: str, limit: int = 14):
    if not text:
        return []
    lines = []
    current = ""
    for char in text:
        current += char
        if len(current) >= limit:
            lines.append(current)
            current = ""
    if current:
        lines.append(current)
    return lines


def draw_speech_bubble(screen: Surface, pos, text: str, font, opacity: float = 1.0) -> None:
    lines = _wrap_text(text)
    if not lines:
        return
    pad = 8
    line_height = font.get_sized_height()
    text_surfaces = [font.render(line, fgcolor=(20, 20, 20))[0] for line in lines]
    width = max(surf.get_width() for surf in text_surfaces) + pad * 2
    height = len(text_surfaces) * line_height + pad * 2
    bubble = Surface((width, height + 6), pygame.SRCALPHA)
    alpha = int(180 * max(0.0, min(1.0, opacity)))
    pygame.draw.rect(bubble, (255, 255, 255, alpha), Rect(0, 0, width, height), border_radius=10)
    pygame.draw.polygon(bubble, (255, 255, 255, alpha), [(width / 2 - 6, height), (width / 2 + 6, height), (width / 2, height + 6)])
    pygame.draw.rect(bubble, (80, 80, 100, alpha), Rect(0, 0, width, height), width=1, border_radius=10)
    for idx, surf in enumerate(text_surfaces):
        bubble.blit(surf, (pad, pad + idx * line_height))
    screen.blit(bubble, (pos[0] - width / 2, pos[1] - bubble.get_height() - 12))


def draw_emotion_aura(screen: Surface, pos, emotion: str, level: float) -> None:
    if level <= 0:
        return
    palette = {
        "기쁨": (255, 220, 120),
        "불안": (140, 180, 255),
        "분노": (255, 110, 110),
        "평온": (120, 200, 180),
    }
    color = palette.get(emotion, (180, 180, 255))
    radius = 22 + 40 * level
    aura_surface = Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    for layer in range(6):
        ratio = 1 - layer / 6
        alpha = int(80 * level * ratio)
        current_radius = int(radius * ratio)
        pygame.draw.circle(aura_surface, (*color, alpha), (radius, radius), current_radius)
    screen.blit(aura_surface, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_ADD)

