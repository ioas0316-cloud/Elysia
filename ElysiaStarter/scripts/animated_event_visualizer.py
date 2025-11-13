import argparse
import json
import os
import sys
import time
import random
from typing import List, Tuple, Dict, Any, Optional

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

# --- Animation System for Events ---
class Animation:
    def __init__(self, duration: float):
        self.start_time = time.time()
        self.duration = duration
        self.is_finished = False

    def update(self):
        if time.time() - self.start_time > self.duration:
            self.is_finished = True

class Lunge(Animation):
    def __init__(self, start_pos, end_pos, duration=0.3):
        super().__init__(duration)
        self.start_pos = start_pos
        self.end_pos = end_pos

    def get_current_pos(self) -> Tuple[float, float]:
        elapsed = time.time() - self.start_time
        progress = min(1.0, elapsed / self.duration)
        # Go to target and back
        if progress < 0.5:
            p = progress * 2
        else:
            p = (1.0 - progress) * 2

        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * p
        y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * p
        return x, y

class HitFlash(Animation):
    def __init__(self, pos: Tuple[float, float], duration=0.25):
        super().__init__(duration)
        self.pos = pos

class LightningBolt(Animation):
    def __init__(self, pos: Tuple[float, float], duration=0.35):
        super().__init__(duration)
        self.pos = pos

class FocusPulse(Animation):
    def __init__(self, pos: Tuple[float, float], duration=1.0):
        super().__init__(duration)
        self.pos = pos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=1024, help='window size (square)')
    ap.add_argument('--fps', type=int, default=60)
    args = ap.parse_args()

    cfg = load_cfg()
    W, H = cfg.get('world', {}).get('grid', [256, 256])

    # --- Initialize Core Components ---
    mock_kg_manager = KGManager()
    mock_wave_mechanics = WaveMechanics(mock_kg_manager)
    world = World(primordial_dna={'instinct': 'observe'}, wave_mechanics=mock_wave_mechanics)

    # Populate the world
    world.add_cell('human_1', properties={'label': 'human', 'element_type': 'animal', 'culture': 'wuxia', 'gender': 'male', 'vitality': 7, 'wisdom': 8, 'strength': 12})
    world.add_cell('human_2', properties={'label': 'human', 'element_type': 'animal', 'culture': 'knight', 'gender': 'female', 'vitality': 8, 'wisdom': 7, 'strength': 10})
    world.add_cell('plant_1', properties={'label': 'tree', 'element_type': 'life'})
    world.add_cell('wolf_1', properties={'label': 'wolf', 'element_type': 'animal', 'diet': 'carnivore', 'strength': 8})
    world.add_connection('wolf_1', 'human_2', 0.1) # Wolf can attack human

    pygame.init()
    screen = pygame.display.set_mode((args.size, args.size))
    pygame.display.set_caption('Elysia\'s Animated World')
    font = get_font(16)
    clock = pygame.time.Clock()

    running = True
    scale, pan_x, pan_y = 1.0, 0.0, 0.0
    dragging, last_mouse = False, (0, 0)

    # --- Event and Animation Management ---
    event_log_path = world.event_logger.log_file_path
    last_log_pos = 0
    animations: Dict[str, Animation] = {} # cell_id -> Animation object
    dying_cells: Dict[str, float] = {} # cell_id -> death_timestamp
    impact_anims: List[Animation] = [] # position-based flashes/bolts/pulses
    event_ticker: List[Tuple[float, str]] = [] # (time, text)
    cinematic_focus = True

    while running:
        dt = clock.tick(args.fps) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_q):
                running = False
            # Handle zoom and pan
            if e.type == pygame.KEYDOWN and e.key == pygame.K_c:
                cinematic_focus = not cinematic_focus
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

        world.run_simulation_step()

        # --- Process World Events for Animation ---
        try:
            with open(event_log_path, 'r', encoding='utf-8') as f:
                f.seek(last_log_pos)
                new_lines = f.readlines()
                last_log_pos = f.tell()

            for line in new_lines:
                event = json.loads(line)
                event_type = event.get('event_type')
                data = event.get('data', {})

                if event_type == 'EAT' and 'actor_id' in data and 'target_id' in data:
                    actor_idx = world.id_to_idx.get(data['actor_id'])
                    target_idx = world.id_to_idx.get(data['target_id'])
                    if actor_idx is not None and target_idx is not None:
                        start_pos = world.positions[actor_idx]
                        end_pos = world.positions[target_idx]
                        animations[data['actor_id']] = Lunge(start_pos, end_pos)
                        impact_anims.append(HitFlash(tuple(end_pos[:2])))
                        event_ticker.append((time.time(), f"{data['actor_id']} eats {data['target_id']}"))

                elif event_type == 'DEATH' and 'cell_id' in data:
                    dying_cells[data['cell_id']] = time.time()
                    idx = world.id_to_idx.get(data['cell_id'])
                    if idx is not None:
                        pos = world.positions[idx]
                        impact_anims.append(FocusPulse(tuple(pos[:2])))
                    event_ticker.append((time.time(), f"{data['cell_id']} died"))

                elif event_type == 'LIGHTNING_STRIKE' and 'cell_id' in data:
                    idx = world.id_to_idx.get(data['cell_id'])
                    if idx is not None:
                        pos = world.positions[idx]
                        impact_anims.append(LightningBolt(tuple(pos[:2])))
                        impact_anims.append(HitFlash(tuple(pos[:2])))
                        event_ticker.append((time.time(), f"⚡ lightning struck {data['cell_id']}"))

        except (IOError, json.JSONDecodeError):
            pass # File might not exist yet or be empty


        screen.fill((30, 30, 40))

        base = min(screen.get_width() / W, screen.get_height() / H)
        s = max(0.5, min(8.0, scale)) * base
        cx, cy = (screen.get_width() - int(W * s)) // 2, (screen.get_height() - int(H * s)) // 2

        def w2s(px: float, py: float) -> Tuple[int, int]:
            return (int(cx - pan_x + px * s), int(cy - pan_y + py * s))

        # --- Update animations and draw ---
        finished_anims = []
        for cell_id, anim in animations.items():
            anim.update()
            if anim.is_finished:
                finished_anims.append(cell_id)
        for cell_id in finished_anims:
            del animations[cell_id]

        # Update position-based impact animations
        new_impacts: List[Animation] = []
        for anim in impact_anims:
            anim.update()
            if not anim.is_finished:
                new_impacts.append(anim)
        impact_anims = new_impacts

        # Optional subtle cinematic focus: pan slightly towards recent impact
        if cinematic_focus and impact_anims:
            # take the last impact as the focus
            last = impact_anims[-1]
            fx, fy = last.pos
            # project into screen coords to nudge pan toward event
            ex, ey = int(cx - pan_x + fx * s), int(cy - pan_y + fy * s)
            # nudge pan by small fraction toward keeping event near center
            target_px, target_py = screen.get_width()//2, screen.get_height()//2
            pan_x = pan_x * 0.9 + (ex - target_px) * 0.1
            pan_y = pan_y * 0.9 + (ey - target_py) * 0.1

        # Draw cells
        for i, cell_id in enumerate(world.cell_ids):
            # Skip drawing if the cell is in the process of a death animation
            if cell_id in dying_cells and time.time() - dying_cells[cell_id] > 1.0:
                continue

            if not world.is_alive_mask[i] and cell_id not in dying_cells:
                continue

            pos = world.positions[i]
            if cell_id in animations:
                anim = animations[cell_id]
                if isinstance(anim, Lunge):
                    pos = anim.get_current_pos()

            sx, sy = w2s(pos[0], pos[1])

            size = 5
            color = (150, 150, 150)
            if world.element_types[i] == 'animal': color = (200, 220, 255)
            elif world.element_types[i] == 'life': color, size = (60, 200, 90), 4
            if world.culture[i] == 'wuxia': color = (220, 100, 100)
            elif world.culture[i] == 'knight': color = (100, 150, 220)

            # --- Death Fade Animation ---
            alpha = 255
            if cell_id in dying_cells:
                elapsed = time.time() - dying_cells[cell_id]
                alpha = int(max(0, 255 * (1.0 - elapsed / 1.0)))

            temp_surf = pygame.Surface((size*2+20, size*2+16), pygame.SRCALPHA)
            # Body
            pygame.draw.circle(temp_surf, color + (alpha,), (size+2, size+2), size)
            if world.is_injured[i]:
                 pygame.draw.circle(temp_surf, (255, 0, 0, alpha), (size+2, size+2), size + 2, 1)
            # Health bar (top)
            if world.max_hp[i] > 0:
                hp_ratio = max(0.0, min(1.0, float(world.hp[i] / world.max_hp[i])))
                bar_w, bar_h = 18, 3
                bx, by = (size+2) - bar_w//2, max(0, (size+2) - (size+6))
                pygame.draw.rect(temp_surf, (60,60,70,160), pygame.Rect(bx, by, bar_w, bar_h), border_radius=2)
                pygame.draw.rect(temp_surf, (60,200,90,220), pygame.Rect(bx, by, int(bar_w*hp_ratio), bar_h), border_radius=2)
            # Hunger bar (bottom)
            hunger_ratio = max(0.0, min(1.0, float(world.hunger[i] / 100.0))) if world.hunger.size>i else 0.0
            bar_w2, bar_h2 = 18, 3
            bx2, by2 = (size+2) - bar_w2//2, (size+2) + (size+6)
            pygame.draw.rect(temp_surf, (60,60,70,160), pygame.Rect(bx2, by2, bar_w2, bar_h2), border_radius=2)
            pygame.draw.rect(temp_surf, (220,190,70,220), pygame.Rect(bx2, by2, int(bar_w2*hunger_ratio), bar_h2), border_radius=2)
            # Action icon: show simple indicator if anim exists
            if cell_id in animations and isinstance(animations[cell_id], Lunge):
                # small forward triangle above head
                px, py = size+2, max(0, (size+2) - (size+10))
                points = [(px, py-3), (px-4, py+3), (px+4, py+3)]
                pygame.draw.polygon(temp_surf, (255,120,120,alpha), points)

            screen.blit(temp_surf, (sx - size - 2 - 10, sy - size - 2 - 8))

        # Draw aim lines for lunges
        for cell_id, anim in animations.items():
            if isinstance(anim, Lunge):
                csx, csy = w2s(anim.start_pos[0], anim.start_pos[1])
                tsx, tsy = w2s(anim.end_pos[0], anim.end_pos[1])
                pygame.draw.line(screen, (255,120,120), (csx, csy), (tsx, tsy), 1)

        # Impact animations (lightning bolt, hit flash, focus pulse)
        for anim in impact_anims:
            if isinstance(anim, LightningBolt):
                # draw jagged bolt
                lx, ly = w2s(anim.pos[0], anim.pos[1])
                bolt = pygame.Surface((40, 60), pygame.SRCALPHA)
                pts = [(20,0),(15,15),(25,15),(18,30),(30,30),(22,50),(28,50),(24,60)]
                pygame.draw.lines(bolt, (240,240,120,220), False, pts, 3)
                screen.blit(bolt, (lx-20, ly-30), special_flags=pygame.BLEND_ADD)
            elif isinstance(anim, HitFlash):
                fx, fy = w2s(anim.pos[0], anim.pos[1])
                r = 12
                surf = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (255,80,80,180), (r,r), r, 2)
                screen.blit(surf, (fx-r, fy-r))
            elif isinstance(anim, FocusPulse):
                fx, fy = w2s(anim.pos[0], anim.pos[1])
                t = min(1.0, (time.time()-anim.start_time)/anim.duration)
                rr = int(20 + 40*t)
                a = int(180*(1.0-t))
                surf = pygame.Surface((rr*2, rr*2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (120,220,255,a), (rr,rr), rr, 2)
                screen.blit(surf, (fx-rr, fy-rr))

        # Day/Night tint overlay
        day_phase = (world.time_step % world.day_length) / float(max(1, world.day_length)) if getattr(world, 'day_length', None) else 0.0
        if day_phase > 0.5: # night half
            night_alpha = int(120 * (day_phase-0.5) * 2)
            tint = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            tint.fill((10,10,30, night_alpha))
            screen.blit(tint, (0,0))

        # HUD
        hh, mm = int(world.time_step // 60), int(world.time_step % 60)
        hud_lines = [f"Time {hh:02d}:{mm:02d}", f"Population {np.sum(world.is_alive_mask)}"]
        for i, tline in enumerate(hud_lines):
            surf, _ = font.render(tline, fgcolor=(235, 235, 245))
            screen.blit(surf, (screen.get_width() - surf.get_width() - 10, 10 + i * (font.get_sized_height() + 2)))

        # Event ticker (bottom-left)
        # keep last 6 entries within 6 seconds
        now = time.time()
        event_ticker = [(t,msg) for (t,msg) in event_ticker if now - t < 6.0]
        for i, (_, msg) in enumerate(event_ticker[-6:]):
            alpha = int(220 * (1.0 - (now - event_ticker[-6:][i][0]) / 6.0))
            surf, _ = font.render(msg, fgcolor=(235,235,245,))
            panel = pygame.Surface((surf.get_width()+10, surf.get_height()+4), pygame.SRCALPHA)
            panel.fill((0,0,0,100))
            panel.blit(surf, (6,2))
            screen.blit(panel, (10, screen.get_height()- (i+1)*(surf.get_height()+8) - 10))


        pygame.display.flip()

    pygame.quit(); sys.exit()


if __name__ == '__main__':
    main()
