import pygame

from core.selection import RectArea, SelectionManager


def draw_drag_rectangle(screen: pygame.Surface, rect: RectArea | None) -> None:
    if not rect:
        return
    normalized = rect.normalized()
    width = normalized.x2 - normalized.x1
    height = normalized.y2 - normalized.y1
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    overlay.fill((60, 220, 120, 70))
    screen.blit(overlay, (normalized.x1, normalized.y1))
    pygame.draw.rect(
        screen,
        (80, 250, 160),
        pygame.Rect(normalized.x1, normalized.y1, width, height),
        width=2,
    )


def draw_selection_highlight(screen: pygame.Surface, selection: SelectionManager) -> None:
    clip = screen.get_rect()
    for agent in selection.get_selected():
        x, y, w, h = agent.get_bbox()
        if not clip.colliderect(pygame.Rect(x, y, w, h)):
            continue
        pygame.draw.rect(
            screen,
            (48, 255, 180),
            pygame.Rect(x, y, w, h),
            width=2,
        )
