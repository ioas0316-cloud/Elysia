import argparse
import json
import os
import sys
import time
import random
from typing import List, Tuple

import numpy as np
import pygame
import yaml


# Ensure project root on sys.path so outer 'ElysiaStarter' is used
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)
_PROJ_ROOT = os.path.dirname(_PKG_ROOT)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from ElysiaStarter.core.cell_world import CellWorld
from ElysiaStarter.core.biome import classify_biome
from ElysiaStarter.ui.view_state import load_view_state, save_view_state
from ElysiaStarter.ui.layer_panel import handle_layer_keys, draw_layer_hud, draw_layer_panel
from ElysiaStarter.ui.fonts import get_font
from ElysiaStarter.ui.layers import LAYERS
from ElysiaStarter.ui.render_overlays import draw_speech_bubble, draw_emotion_aura


def load_cfg():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(here, 'config', 'runtime.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def biome_to_rgb(biome: np.ndarray) -> np.ndarray:
    colors = np.array([
        [30, 30, 40],    # 0
        [60, 120, 200],  # 1 water
        [180, 180, 140], # 2 sand
        [110, 170, 80],  # 3 grass
        [40, 110, 50],   # 4 forest
        [140, 120, 120], # 5 rock
    ], dtype=np.uint8)
    idx = np.clip(biome.astype(np.int32), 0, len(colors) - 1)
    return colors[idx]


# ======== Minimal Civilization Entities ========

SEX_CHOICES = ['M', 'F']
AGE_STAGES = ['child', 'adult', 'elder']
EMOTIONS = ['calm', 'joy', 'nervous', 'anger']
TEMPLATES = {
    'greet': ["안녕하세요", "만나서 반가워요"],
}


class AgentEnt:
    __slots__ = (
        'id', 'name', 'sex', 'age', 'x', 'y', 'emotion', 'emotion_level',
        'hunger', 'last_utter', 'utter_time', 'utter_ttl', 'last_reason',
        'schedule')

    def __init__(self, idx: int, x: float, y: float):
        self.id = idx
        self.name = f"사람-{idx:02d}"
        self.sex = random.choice(SEX_CHOICES)
        self.age = random.choices(AGE_STAGES, weights=[0.2, 0.6, 0.2])[0]
        self.x, self.y = x, y
        self.emotion = random.choice(EMOTIONS)
        self.emotion_level = random.random()
        self.hunger = random.uniform(0.2, 0.6)
        self.last_utter = ""
        self.utter_time = 0.0
        self.utter_ttl = 2.0
        self.last_reason = ""
        self.schedule = [("work", 600, 1020), ("rest", 1020, 1320)]

    def speak(self, text: str, now: float, reason: str = ""):
        self.last_utter = text
        self.utter_time = now
        if reason:
            self.last_reason = reason

    def can_speak(self, now: float) -> bool:
        return (now - self.utter_time) > 5.0

    def radius(self) -> int:
        return 6 if self.age == 'child' else (8 if self.age == 'adult' else 7)


class Flora:
    def __init__(self, x: int, y: int):
        self.kind = 'berry_bush'
        self.x, self.y = x, y
        self.energy = 1.0


class Fauna:
    def __init__(self, x: float, y: float):
        self.kind = 'deer'
        self.x, self.y = x, y
        self.stress = 0.0


def create_agents(n: int, W: int, H: int) -> List[AgentEnt]:
    return [AgentEnt(i, random.uniform(0, W - 1), random.uniform(0, H - 1)) for i in range(n)]


def spawn_ecology(W: int, H: int, water: np.ndarray) -> Tuple[List[Flora], List[Fauna]]:
    flora, fauna = [], []
    for _ in range(40):
        x = random.randint(0, W - 1)
        y = random.randint(0, H - 1)
        if water[y, x] == 1 or random.random() < 0.15:
            flora.append(Flora(x, y))
    for _ in range(12):
        x = random.randint(0, W - 1)
        y = random.randint(0, H - 1)
        fauna.append(Fauna(x, y))
    return flora, fauna


def draw_help(screen: pygame.Surface, font, y_start: int) -> int:
    lines = [
        "[H] Help  Q: Quit",
        "Move: WASD  Zoom: Wheel  Pan: Middle Drag",
        "Layers: A/S/F/Z/W  Select: Left Click",
    ]
    pad = 10
    line_h = max(18, font.get_sized_height() + 2)
    h = pad + len(lines) * line_h + pad
    surfs = [font.render(t, fgcolor=(220, 220, 230))[0] for t in lines]
    w = max(260, max(s.get_width() for s in surfs) + 20)
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 150))
    y = pad
    for surf in surfs:
        panel.blit(surf, (10, y))
        y += line_h
    screen_h = screen.get_height()
    y_pos = y_start + 10
    if y_pos + h > screen_h - 10:
        y_pos = screen_h - h - 10
    screen.blit(panel, (10, y_pos))
    return y_pos + h


def ensure_dir(p: str):
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=1024, help='window size (square)')
    ap.add_argument('--fps', type=int, default=30)
    args = ap.parse_args()

    cfg = load_cfg()
    W, H = cfg.get('world', {}).get('grid', [256, 256])

    world = CellWorld(W, H)

    pygame.init()
    screen = pygame.display.set_mode((args.size, args.size))
    pygame.display.set_caption('Elysia Timeline (minimal)')
    font = get_font(16)
    clock = pygame.time.Clock()

    # Load persisted view state
    state_path = os.path.join(_PKG_ROOT, 'saves', 'viewer_state.json')
    try:
        _ = load_view_state(state_path)
    except Exception:
        pass

    running = True
    # Zoom/Pan state
    scale = 1.0
    pan_x = 0.0
    pan_y = 0.0
    dragging = False
    last_mouse = (0, 0)
    last_print = time.time()
    # Civilization state
    agents: List[AgentEnt] = create_agents(25, W, H)
    flora: List[Flora] = []
    fauna: List[Fauna] = []
    # time (minutes in a day 0..1440)
    sim_min = 480  # 08:00
    minutes_per_sec = 2.0
    # metrics
    knowledge_points = 0
    events_count_10m = 0
    last_metrics_log = time.time()
    metrics_path = os.path.join(_PKG_ROOT, 'saves', 'metrics.jsonl')
    ensure_dir(metrics_path)
    # selection
    selected_ids: set[int] = set()
    show_only_selected = False

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            handle_layer_keys(e)
            if e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_t:
                show_only_selected = not show_only_selected
            if e.type == pygame.KEYDOWN and e.key == pygame.K_c:
                selected_ids.clear()
            # Zoom (wheel)
            if e.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                base = min(screen.get_width() / W, screen.get_height() / H)
                s_old = base * scale
                scale = max(0.5, min(4.0, scale * (1.1 if e.y > 0 else 0.9)))
                s_new = base * scale
                # Keep cursor position during zoom
                u = mx + pan_x
                v = my + pan_y
                pan_x = (u / s_old) * s_new - mx
                pan_y = (v / s_old) * s_new - my
            # Start panning (middle button)
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 2:
                dragging = True
                last_mouse = e.pos
            if e.type == pygame.MOUSEBUTTONUP and e.button == 2:
                dragging = False
            if e.type == pygame.MOUSEMOTION and dragging:
                x, y = e.pos
                lx, ly = last_mouse
                dx, dy = x - lx, y - ly
                pan_x += -dx
                pan_y += -dy
                last_mouse = (x, y)
            # simple click-select nearest agent
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                mx, my = e.pos
                base = min(screen.get_width() / W, screen.get_height() / H)
                s = max(0.5, min(4.0, scale)) * base
                cx = (screen.get_width() - int(W * s)) // 2
                cy = (screen.get_height() - int(H * s)) // 2
                nearest = None
                best_d2 = 1e18
                for a in agents:
                    sx = cx - pan_x + a.x * s
                    sy = cy - pan_y + a.y * s
                    d2 = (sx - mx) ** 2 + (sy - my) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        nearest = a
                mods = pygame.key.get_mods()
                if nearest and best_d2 < (18 ** 2):
                    if mods & pygame.KMOD_CTRL:
                        if nearest.id in selected_ids:
                            selected_ids.remove(nearest.id)
                    elif mods & pygame.KMOD_SHIFT:
                        selected_ids.add(nearest.id)
                    else:
                        selected_ids = {nearest.id}

        # Update world and classify
        world.update_fields()
        biome = classify_biome(world.height, world.moisture, world.temp)
        # spawn ecology once
        if not flora:
            flora, fauna = spawn_ecology(W, H, (biome == 1).astype(np.uint8))

        # time update
        dt = clock.get_time() / 1000.0
        sim_min = (sim_min + minutes_per_sec * dt) % 1440.0
        hh = int(sim_min // 60)
        mm = int(sim_min % 60)

        # Render biome as image
        rgb = biome_to_rgb(biome)  # (H,W,3)
        base_surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))  # (W,H)
        base = min(screen.get_width() / W, screen.get_height() / H)
        s = max(0.5, min(4.0, scale)) * base
        scaled_w = max(1, int(W * s))
        scaled_h = max(1, int(H * s))
        surf = pygame.transform.smoothscale(base_surf, (scaled_w, scaled_h))
        # Center image and apply pan
        cx = (screen.get_width() - scaled_w) // 2
        cy = (screen.get_height() - scaled_h) // 2
        screen.blit(surf, (cx - pan_x, cy - pan_y))

        # helpers: world->screen
        def w2s(px: float, py: float) -> Tuple[int, int]:
            return (int(cx - pan_x + px * s), int(cy - pan_y + py * s))

        # simple behaviors + interactions
        for a in agents:
            a.hunger = min(1.0, a.hunger + 0.01 * (dt if 'dt' in locals() else 0.016))
            in_work = any(st <= sim_min <= en for (k, st, en) in a.schedule if k == 'work')
            a.emotion = 'joy' if in_work else 'calm'
            a.emotion_level = 0.2 + (0.6 if in_work else 0.3)

        # interactions (random short greetings when near)
        now = time.time()
        for i in range(len(agents)):
            ai = agents[i]
            if not ai.can_speak(now):
                continue
            j = random.randrange(len(agents))
            aj = agents[j]
            if ai is aj:
                continue
            if (ai.x - aj.x) ** 2 + (ai.y - aj.y) ** 2 < (18 ** 2):
                ai.speak(random.choice(TEMPLATES['greet']), now, reason='social')

        # draw flora/fauna
        if LAYERS.get('flora', True):
            for f in flora:
                sx, sy = w2s(f.x, f.y)
                pygame.draw.circle(screen, (60, 200, 90), (sx, sy), 4)
        if LAYERS.get('fauna', True):
            for f in fauna:
                sx, sy = w2s(f.x, f.y)
                pygame.draw.circle(screen, (200, 180, 120), (sx, sy), 5, 1)

        # draw agents
        draw_list = agents if not show_only_selected else [a for a in agents if a.id in selected_ids]
        if LAYERS.get('agents', True):
            for a in draw_list:
                sx, sy = w2s(a.x, a.y)
                draw_emotion_aura(screen, (sx, sy), a.emotion, a.emotion_level)
                color = (200, 220, 255) if a.sex == 'M' else (255, 180, 200)
                pygame.draw.circle(screen, color, (sx, sy), a.radius())
                if a.last_utter and (now - a.utter_time) < a.utter_ttl + 0.5:
                    age = now - a.utter_time
                    opacity = 1.0 if age < a.utter_ttl else max(0.0, 1.0 - (age - a.utter_ttl) / 0.5)
                    draw_speech_bubble(screen, (sx, sy - a.radius() - 6), a.last_utter, font, opacity)
                if a.id in selected_ids:
                    pygame.draw.circle(screen, (80, 255, 180), (sx, sy), a.radius() + 2, 1)

        # HUD: help + layer indicators + metrics
        draw_layer_hud(screen)
        draw_layer_panel(screen)
        _ = draw_help(screen, font, 10)
        hud_lines = [
            f"Time {hh:02d}:{mm:02d}",
            f"Pop {len(agents)}  Sel {len(selected_ids)}",
            f"Food {sum(1 for f in flora if f.energy>0)}  KP {knowledge_points}",
        ]
        y0 = 10
        for i, tline in enumerate(hud_lines):
            surf, _ = font.render(tline, fgcolor=(235, 235, 245))
            screen.blit(surf, (10, y0 + i * (font.get_sized_height() + 2)))

        # metrics logging (best-effort)
        if time.time() - last_metrics_log > 10.0:
            last_metrics_log = time.time()
            try:
                with open(metrics_path, 'a', encoding='utf-8') as f:
                    rec = {
                        'ts': time.time(),
                        'time_hhmm': f"{hh:02d}:{mm:02d}",
                        'pop': len(agents),
                        'food': sum(1 for f in flora if f.energy>0),
                        'knowledge_points': knowledge_points,
                        'events_10m': events_count_10m,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                events_count_10m = 0
            except Exception:
                pass

        pygame.display.flip()
        clock.tick(args.fps)

        if time.time() - last_print > 1.0:
            last_print = time.time()

    try:
        save_view_state(state_path, {"pos": [0.0, 0.0], "zoom": 1.0}, False)
    except Exception:
        pass
    pygame.quit(); sys.exit()


if __name__ == '__main__':
    main()

