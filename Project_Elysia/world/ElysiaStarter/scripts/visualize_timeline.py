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


# Ensure project root on sys.path for cross-project imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Repo root is two levels up from this script (ElysiaStarter/scripts)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# --- Import the REAL World and Sensory Cortex ---
from Project_Sophia.core.world import World
from Project_Sophia.wave_mechanics import WaveMechanics
from Project_Mirror.sensory_cortex import SensoryCortex
from Project_Sophia.value_cortex import ValueCortex
from tools.kg_manager import KGManager

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

def ensure_dir(p: str):
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    except Exception:
        pass

# --- Particle System for Events ---
class Particle:
    def __init__(self, x, y, color, max_life=1.0):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.life = max_life
        self.max_life = max_life
        self.color = color

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt

    def draw(self, surface, w2s_func):
        if self.life > 0:
            sx, sy = w2s_func(self.x, self.y)
            alpha = int(255 * (self.life / self.max_life))
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, self.color + (alpha,), (2, 2), 2)
            surface.blit(temp_surf, (sx - 2, sy - 2), special_flags=pygame.BLEND_RGBA_ADD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=1024, help='window size (square)')
    ap.add_argument('--fps', type=int, default=30)
    args = ap.parse_args()

    cfg = load_cfg()
    W, H = cfg.get('world', {}).get('grid', [256, 256])

    # --- Initialize Core Components ---
    mock_kg_manager = KGManager()
    mock_wave_mechanics = WaveMechanics(mock_kg_manager)
    value_cortex = ValueCortex(kg_path='data/kg.json') # SensoryCortex needs this
    sensory_cortex = SensoryCortex(value_cortex)
    world = World(primordial_dna={'instinct': 'observe'}, wave_mechanics=mock_wave_mechanics)

    # Populate the world
    world.add_cell('human_1', properties={'label': 'human', 'element_type': 'animal', 'culture': 'wuxia', 'gender': 'male', 'vitality': 7, 'wisdom': 8})
    world.add_cell('human_2', properties={'label': 'human', 'element_type': 'animal', 'culture': 'knight', 'gender': 'female', 'vitality': 8, 'wisdom': 7})
    world.add_cell('plant_1', properties={'label': 'tree', 'element_type': 'life'})
    world.add_cell('wolf_1', properties={'label': 'wolf', 'element_type': 'animal', 'diet': 'carnivore'})

    pygame.init()
    screen = pygame.display.set_mode((args.size, args.size))
    pygame.display.set_caption('Elysia\'s Inner World')
    font = get_font(16)
    clock = pygame.time.Clock()

    running = True
    scale, pan_x, pan_y = 1.0, 0.0, 0.0
    dragging, last_mouse = False, (0, 0)
    particles: List[Particle] = []

    while running:
        dt = clock.tick(args.fps) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_q):
                running = False
            handle_layer_keys(e)
            # Handle zoom and pan
            if e.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                base = min(screen.get_width() / W, screen.get_height() / H)
                s_old = base * scale
                scale = max(0.5, min(8.0, scale * (1.1 if e.y > 0 else 0.9)))
                s_new = base * scale
                u,v = mx + pan_x, my + pan_y
                pan_x, pan_y = (u / s_old) * s_new - mx, (v / s_old) * s_new - my
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 2:
                dragging, last_mouse = True, e.pos
            if e.type == pygame.MOUSEBUTTONUP and e.button == 2:
                dragging = False
            if e.type == pygame.MOUSEMOTION and dragging:
                dx, dy = e.pos[0] - last_mouse[0], e.pos[1] - last_mouse[1]
                pan_x, pan_y = pan_x - dx, pan_y - dy
                last_mouse = e.pos

        # --- World Simulation and Event Handling ---
        dead_before = np.where(~world.is_alive_mask)[0]
        world.run_simulation_step()
        dead_after = np.where(~world.is_alive_mask)[0]
        newly_dead_indices = np.setdiff1d(dead_after, dead_before)

        # --- Create Particles for Events ---
        for dead_idx in newly_dead_indices:
            pos = world.positions[dead_idx]
            palette = sensory_cortex._get_color_palette("death")
            for _ in range(30):
                particles.append(Particle(pos[0], pos[1], random.choice(palette), max_life=1.5))

        screen.fill((30, 30, 40))

        base = min(screen.get_width() / W, screen.get_height() / H)
        s = max(0.5, min(8.0, scale)) * base
        cx, cy = (screen.get_width() - int(W * s)) // 2, (screen.get_height() - int(H * s)) // 2

        def w2s(px: float, py: float) -> Tuple[int, int]:
            return (int(cx - pan_x + px * s), int(cy - pan_y + py * s))

        if LAYERS.get('agents', True):
            for i, cell_id in enumerate(world.cell_ids):
                if not world.is_alive_mask[i]:
                    continue
                sx, sy = w2s(world.positions[i][0], world.positions[i][1])
                size = 5
                color = (150, 150, 150)
                if world.element_types[i] == 'animal': color = (200, 220, 255)
                elif world.element_types[i] == 'life': color, size = (60, 200, 90), 4
                if world.culture[i] == 'wuxia': color = (220, 100, 100)
                elif world.culture[i] == 'knight': color = (100, 150, 220)

                draw_emotion_aura(screen, (sx, sy), world.emotions[i], 0.7)
                pygame.draw.circle(screen, color, (sx, sy), size)
                if world.is_injured[i]:
                    pygame.draw.circle(screen, (255, 0, 0), (sx, sy), size + 2, 1)

        # Update and draw particles
        for p in particles:
            p.update(dt)
            p.draw(screen, w2s)
        particles = [p for p in particles if p.life > 0]

        draw_layer_hud(screen)
        hh, mm = int(world.time_step // 60), int(world.time_step % 60)
        hud_lines = [f"Time {hh:02d}:{mm:02d}", f"Population {np.sum(world.is_alive_mask)}"]
        for i, tline in enumerate(hud_lines):
            surf, _ = font.render(tline, fgcolor=(235, 235, 245))
            screen.blit(surf, (screen.get_width() - surf.get_width() - 10, 10 + i * (font.get_sized_height() + 2)))

        pygame.display.flip()

    pygame.quit(); sys.exit()

if __name__ == '__main__':
    main()
