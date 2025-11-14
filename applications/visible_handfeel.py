import os
import sys
import random
import time

import pygame

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app_core.agent import Agent
from app_core.events import meeting, task_failure, task_success
from app_core.selection import SelectionManager
from app_core.state import UIState
from ui.fonts import get_font
from ui.input import InputController
from ui.render_overlays import draw_emotion_aura, draw_speech_bubble
from ui.render_selection import draw_drag_rectangle, draw_selection_highlight

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
BACKGROUND_COLOR = (16, 18, 25)
AGENT_COLORS = [
    (225, 180, 120),
    (180, 210, 255),
    (230, 120, 160),
    (155, 255, 190),
]
EMOTIONS = ["?됱삩", "湲곗겏", "遺덉븞", "遺꾨끂"]
ROLES = ["嫄댁텞媛", "?섑샇??, "?먯깋??, "吏?섏옄", "?뺤같??, "吏吏??]
NAMES = ["吏??, "?섎뒛", "?먮뵒", "誘몃Ⅴ", "蹂?, "蹂대━", "?뚮━", "?섎엺", "??, "?ш린"]


def create_agents(count: int, width: int, height: int) -> list[Agent]:
    random.seed(42)
    agents: list[Agent] = []
    for idx in range(count):
        name = random.choice(NAMES)
        role = random.choice(ROLES)
        x = random.uniform(80, width - 80)
        y = random.uniform(120, height - 120)
        color = AGENT_COLORS[idx % len(AGENT_COLORS)]
        agent = Agent(
            id=idx,
            name=f"{name}-{idx:02d}",
            role=role,
            x=x,
            y=y,
            color=color,
        )
        emotion = random.choice(EMOTIONS)
        agent.set_emotion(emotion, random.random())
        agents.append(agent)
    return agents


def _pick_target(selection: SelectionManager, agents: list[Agent]) -> Agent | None:
    selected = list(selection.get_selected())
    if selected:
        return selected[0]
    return random.choice(agents) if agents else None


def _draw_hud(screen: pygame.Surface, font: "pygame.freetype.Font", state: UIState, world_time: float, agents: list[Agent], selection: SelectionManager) -> None:
    grouped = [
        f"Time: {world_time:05.1f}s",
        f"Pop: {len(agents)}  Sel: {len(selection.selected_ids)}",
        f"View: {'SELECTED' if state.show_only_selected else 'ALL'}",
    ]
    base_y = 10
    for idx, line in enumerate(grouped):
        surf, _ = font.render(line, fgcolor=(235, 235, 245))
        screen.blit(surf, (10, base_y + idx * (font.get_sized_height() + 4)))


def _draw_tooltip(
    screen: pygame.Surface,
    font: "pygame.freetype.Font",
    agent: Agent,
    position: tuple[int, int],
) -> None:
    text = f"{agent.name} / {agent.role} / {agent.emotion}"
    surf, rect = font.render(text, fgcolor=(255, 255, 255))
    padding = 6
    w = surf.get_width() + padding * 2
    h = surf.get_height() + padding * 2
    tooltip = pygame.Surface((w, h), pygame.SRCALPHA)
    tooltip.fill((10, 10, 15, 200))
    pygame.draw.rect(tooltip, (60, 180, 140), tooltip.get_rect(), width=1)
    tooltip.blit(surf, (padding, padding))
    screen.blit(tooltip, (position[0] + 12, position[1] + 12))


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Elysia Visible Handfeel")
    font = get_font(18)
    clock = pygame.time.Clock()
    agents = create_agents(28, SCREEN_WIDTH, SCREEN_HEIGHT)
    selection = SelectionManager(agents)
    state = UIState()
    controller = InputController(selection, state)
    running = True
    world_time = 0.0

    while running:
        delta = clock.tick(60) / 1000.0
        world_time += delta
        now = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            controller.handle_event(event)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    target = _pick_target(selection, agents)
                    if target:
                        task_success(target, now)
                elif event.key == pygame.K_2:
                    target = _pick_target(selection, agents)
                    if target:
                        task_failure(target, now)
                elif event.key == pygame.K_3:
                    target = _pick_target(selection, agents)
                    if target:
                        meeting(target, now)

        drawable = [
            agent
            for agent in agents
            if 0 - agent.radius <= agent.x <= SCREEN_WIDTH + agent.radius
            and 0 - agent.radius <= agent.y <= SCREEN_HEIGHT + agent.radius
        ]
        if state.show_only_selected:
            drawable = [agent for agent in drawable if agent.id in selection.selected_ids]

        screen.fill(BACKGROUND_COLOR)
        for agent in drawable:
            draw_emotion_aura(screen, (agent.x, agent.y), agent.emotion, agent.emotion_level)
            pygame.draw.circle(
                screen,
                agent.color,
                (int(agent.x), int(agent.y)),
                int(agent.radius),
            )
            if agent.has_speech(now):
                opacity = agent.speech_opacity(now)
                draw_speech_bubble(
                    screen,
                    (agent.x, agent.y - agent.radius),
                    agent.last_utterance,
                    font,
                    opacity=opacity,
                )

        draw_selection_highlight(screen, selection)
        draw_drag_rectangle(screen, controller.get_active_rect())
        _draw_hud(screen, font, state, world_time, agents, selection)

        mouse_pos = pygame.mouse.get_pos()
        tooltip_agent = None
        for agent in selection.get_selected():
            x, y, w, h = agent.get_bbox()
            if x <= mouse_pos[0] <= x + w and y <= mouse_pos[1] <= y + h:
                tooltip_agent = agent
                break
        if tooltip_agent:
            _draw_tooltip(screen, font, tooltip_agent, mouse_pos)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
