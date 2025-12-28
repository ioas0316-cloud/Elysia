import pygame

from app_core.selection import RectArea, SelectionManager
from app_core.state import UIState


class InputController:
    def __init__(self, selection: SelectionManager, state: UIState):
        self.selection = selection
        self.state = state
        self.dragging = False
        self.selection_start: tuple[float, float] | None = None
        self.selection_rect: RectArea | None = None

    def handle_event(self, event: pygame.event.EventType) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.dragging = True
            self.selection_start = event.pos
            self.selection_rect = RectArea(event.pos[0], event.pos[1], event.pos[0], event.pos[1])
        elif event.type == pygame.MOUSEMOTION and self.dragging and self.selection_start:
            x0, y0 = self.selection_start
            x1, y1 = event.pos
            self.selection_rect = RectArea(x0, y0, x1, y1)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging and self.selection_rect:
                mods = pygame.key.get_mods()
                mode = "replace"
                if mods & pygame.KMOD_SHIFT:
                    mode = "add"
                if mods & pygame.KMOD_CTRL:
                    mode = "remove"
                self.selection.select(self.selection_rect, mode=mode)
            self.dragging = False
            self.selection_start = None
            self.selection_rect = None
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:
                self.state.show_only_selected = not self.state.show_only_selected
            elif event.key == pygame.K_c:
                self.selection.clear()

    def get_active_rect(self) -> RectArea | None:
        if self.selection_rect:
            return self.selection_rect.normalized()
        return None
