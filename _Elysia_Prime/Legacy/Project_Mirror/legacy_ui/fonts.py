# [Genesis: 2025-12-02] Purified by Elysia
import pygame.freetype as ft


def get_font(size: int) -> ft.Font:
    """Return a Korean-friendly font, trying system options in priority order."""
    candidates = ["Malgun Gothic", "MalgunGothic", "AppleGothic", "NanumGothic", "Arial Unicode MS", "sans-serif"]
    for name in candidates:
        try:
            font = ft.SysFont(name, size)
            if font:
                return font
        except Exception:
            continue
    return ft.SysFont(None, size)