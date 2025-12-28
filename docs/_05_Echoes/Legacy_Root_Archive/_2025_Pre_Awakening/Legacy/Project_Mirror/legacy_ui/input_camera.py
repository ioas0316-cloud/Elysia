from __future__ import annotations

import pygame
from typing import Optional, Tuple

from .camera import Camera


class CameraController:
    def __init__(self, camera: Camera):
        self.camera = camera
        self._dragging = False
        self._drag_button: Optional[int] = None
        self._last_mouse: Optional[Tuple[int, int]] = None

    def handle_event(self, e: pygame.event.Event) -> None:
        if e.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            self._zoom_at((mx, my), 1.1 if e.y > 0 else 0.9)
        elif e.type == pygame.MOUSEBUTTONDOWN:
            if e.button == 2 or (e.button == 1 and pygame.key.get_mods() & pygame.KMOD_SPACE):
                self._dragging = True
                self._drag_button = e.button
                self._last_mouse = e.pos
        elif e.type == pygame.MOUSEBUTTONUP:
            if self._dragging and e.button == self._drag_button:
                self._dragging = False
                self._drag_button = None
                self._last_mouse = None
        elif e.type == pygame.MOUSEMOTION:
            if self._dragging and self._last_mouse is not None:
                x, y = e.pos
                lx, ly = self._last_mouse
                dx = (lx - x) / self.camera.zoom
                dy = (ly - y) / self.camera.zoom
                self.camera.move(dx, dy)
                self._last_mouse = e.pos
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
            self.camera.pos_x = 0.0
            self.camera.pos_y = 0.0
            self.camera.zoom = 1.0

    def _zoom_at(self, screen_pt, scale):
        # keep cursor world-point stable
        before = self.camera.screen_to_world(screen_pt)
        self.camera.set_zoom(self.camera.zoom * scale)
        after = self.camera.screen_to_world(screen_pt)
        self.camera.move(before[0] - after[0], before[1] - after[1])

