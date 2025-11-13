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
from Project_Elysia.core import persistence as world_persistence


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

    # --- Populate the world (observer-friendly seed) ---
    rng = np.random.default_rng(42)
    def rand_pos():
        return {'x': float(rng.uniform(0, W-1)), 'y': float(rng.uniform(0, H-1)), 'z': 0}

    # Seed a small village of humans
    for i in range(8):
        culture = 'wuxia' if i % 2 == 0 else 'knight'
        world.add_cell(f'human_{i+1}', properties={'label': 'human', 'element_type': 'animal', 'culture': culture, 'gender': 'male' if i%2==0 else 'female', 'vitality': int(rng.integers(7,11)), 'wisdom': int(rng.integers(6,11)), 'strength': int(rng.integers(8,13)), 'position': rand_pos()})

    # Seed wildlife and plants
    for i in range(10):
        world.add_cell(f'plant_{i+1}', properties={'label': 'tree', 'element_type': 'life', 'position': rand_pos()})
    for i in range(5):
        world.add_cell(f'wolf_{i+1}', properties={'label': 'wolf', 'element_type': 'animal', 'diet': 'carnivore', 'strength': int(rng.integers(7,11)), 'position': rand_pos()})
    for i in range(6):
        world.add_cell(f'deer_{i+1}', properties={'label': 'deer', 'element_type': 'animal', 'diet': 'herbivore', 'position': rand_pos()})

    # Light social graph edges to encourage interactions
    ids = list(world.cell_ids)
    animals = [cid for cid in ids if world.element_types[world.id_to_idx[cid]] == 'animal']
    for _ in range(20):
        a,b = rng.choice(animals, size=2, replace=False)
        if a != b:
            try:
                world.add_connection(a, b, float(rng.uniform(0.05, 0.2)))
            except Exception:
                pass

    # Observer-friendly initial conditions: avoid instant deaths
    if world.hunger.size:
        world.hunger[:] = 95.0
    if world.hp.size and world.max_hp.size:
        world.hp[:] = np.maximum(world.hp, world.max_hp * 0.95)
    if getattr(world, 'is_injured', None) is not None and world.is_injured.size:
        world.is_injured[:] = False

    # Calmer default weather for survivability (stormy preset available)
    world.cloud_cover = 0.3
    world.humidity = 0.3
    if hasattr(world, 'day_length'):
        world.day_length = max(120, int(world.day_length)*6)

    pygame.init()
    screen = pygame.display.set_mode((args.size, args.size))
    pygame.display.set_caption('Elysia\'s Animated World')
    font = get_font(16)
    clock = pygame.time.Clock()
    try:
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)
    except Exception:
        pass

    running = True
    scale, pan_x, pan_y = 1.0, 0.0, 0.0
    dragging, last_mouse = False, (0, 0)
    # Real-time paced stepping (steps per second)
    sim_rate = 0.25  # steps per second (very slow by default for observation)
    sim_accum = 0.0
    sim_speed = 1  # kept for compatibility with +/- UI, mapped to sim_rate
    paused = False

    # --- Event and Animation Management ---
    event_log_path = world.event_logger.log_file_path
    last_log_pos = 0
    animations: Dict[str, Animation] = {} # cell_id -> Animation object
    dying_cells: Dict[str, float] = {} # cell_id -> death_timestamp
    impact_anims: List[Animation] = [] # position-based flashes/bolts/pulses
    event_ticker: List[Tuple[float, str]] = [] # (time, text)
    cinematic_focus = False
    show_labels = False
    show_grid = True
    show_terrain = True
    show_threat = False
    selected_id: Optional[str] = None
    trail: List[Tuple[float,float]] = []
    show_help = True  # in-app help overlay
    skill_mode = 'target'
    skill_aoe_radius = 6.0
    divine_mode = False  # 신성력 모드 (커서로 치유/강화)

    def screen_to_world(mx:int, my:int, scale_val:float) -> Tuple[float,float]:
        base_local = min(screen.get_width() / W, screen.get_height() / H)
        s_local = base_local * scale_val
        cx_local = (screen.get_width() - int(W * s_local)) // 2
        cy_local = (screen.get_height() - int(H * s_local)) // 2
        wx = (mx - cx_local + pan_x) / max(1e-6, s_local)
        wy = (my - cy_local + pan_y) / max(1e-6, s_local)
        return float(wx), float(wy)

    def apply_divine_power(wx: float, wy: float, radius_world: float = 6.0) -> int:
        """Heal/inspire agents within radius around world coords (wx, wy)."""
        affected = 0
        if not world.cell_ids:
            return affected
        r2 = radius_world * radius_world
        for i, cid in enumerate(world.cell_ids):
            if i >= world.positions.shape[0]:
                continue
            if not (world.is_alive_mask.size>i and world.is_alive_mask[i]):
                continue
            dx = float(world.positions[i][0]) - wx
            dy = float(world.positions[i][1]) - wy
            if dx*dx + dy*dy <= r2:
                try:
                    world.inject_stimulus(cid, 10.0)
                except Exception:
                    if world.hp.size>i and world.max_hp.size>i:
                        world.hp[i] = min(world.max_hp[i], world.hp[i] + 10.0)
                if world.is_injured.size>i:
                    world.is_injured[i] = False
                if world.max_ki.size>i and world.ki.size>i:
                    world.ki[i] = min(world.max_ki[i], world.ki[i] + 5.0)
                if world.max_mana.size>i and world.mana.size>i:
                    world.mana[i] = min(world.max_mana[i], world.mana[i] + 5.0)
                affected += 1
        return affected

def ui_notify(msg: str):
    try:
        event_ticker.append((time.time(), msg))
    except Exception:
        pass
    
    # Skill casting mode and helpers
    # mode: 'target' (apply to selected unit) or 'aoe' (apply around cursor)
    # Toggle with X key

    def _apply_skill_to_index(skill: str, idx: int) -> None:
        # Q heal: HP+12, clear injury
        # W inspire: Ki/Mana +8
        # E shield: HP+6, clear injury
        # R smite: damage (carnivore priority)
        if skill == 'Q':
            if world.hp.size>idx and world.max_hp.size>idx:
                world.hp[idx] = min(world.max_hp[idx], world.hp[idx] + 12.0)
            if world.is_injured.size>idx:
                world.is_injured[idx] = False
        elif skill == 'W':
            if world.max_ki.size>idx and world.ki.size>idx:
                world.ki[idx] = min(world.max_ki[idx], world.ki[idx] + 8.0)
            if world.max_mana.size>idx and world.mana.size>idx:
                world.mana[idx] = min(world.max_mana[idx], world.mana[idx] + 8.0)
        elif skill == 'E':
            if world.hp.size>idx and world.max_hp.size>idx:
                world.hp[idx] = min(world.max_hp[idx], world.hp[idx] + 6.0)
            if world.is_injured.size>idx:
                world.is_injured[idx] = False
        elif skill == 'R':
            is_carn = (world.diets.size>idx and world.diets[idx] == 'carnivore')
            dmg = 15.0 if is_carn else 6.0
            if world.hp.size>idx:
                world.hp[idx] = max(0.0, world.hp[idx] - dmg)

    def _apply_skill_aoe(skill: str, wx: float, wy: float, radius_world: float) -> int:
        affected = 0
        if not world.cell_ids:
            return affected
        r2 = radius_world * radius_world
        for i, cid in enumerate(world.cell_ids):
            if i >= world.positions.shape[0]:
                continue
            if not (world.is_alive_mask.size>i and world.is_alive_mask[i]):
                continue
            dx = float(world.positions[i][0]) - wx
            dy = float(world.positions[i][1]) - wy
            if dx*dx + dy*dy > r2:
                continue
            _apply_skill_to_index(skill, i)
            affected += 1
        return affected


    # Precompute a simple terrain noise lens (does not touch world state)
    def _gen_noise(w:int, h:int, oct=4):
        base = np.zeros((h,w), dtype=np.float32)
        r = np.random.default_rng(7)
        for k in range(oct):
            sx = max(1, w // (2**(k+2)))
            sy = max(1, h // (2**(k+2)))
            small = r.random((sy,sx)).astype(np.float32)
            up = np.kron(small, np.ones((h//sy, w//sx), dtype=np.float32))
            up = up[:h,:w]
            base += up * (0.6**k)
        base -= base.min(); base /= (base.max() + 1e-6)
        return base

    terrain_noise = _gen_noise(128,128)
    def draw_terrain(surface: pygame.Surface, s: float, cx: int, cy: int, pan_x: float, pan_y: float):
        if not show_terrain:
            return
        # colorize noise into simple biome colors
        arr = (terrain_noise*255).astype(np.uint8)
        palette = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        water = arr < 90
        sand = (arr >= 90) & (arr < 110)
        grass = (arr >= 110) & (arr < 200)
        rock = arr >= 200
        palette[water] = (30,40,70)
        palette[sand] = (170,150,110)
        palette[grass] = (60,110,70)
        palette[rock] = (80,80,80)
        surf = pygame.surfarray.make_surface(np.rot90(palette))
        world_px = (max(1, int(W*s)), max(1, int(H*s)))
        surf = pygame.transform.smoothscale(surf, world_px)
        topleft = (int(cx - pan_x), int(cy - pan_y))
        surface.blit(surf, topleft)
        if show_grid:
            grid = pygame.Surface(world_px, pygame.SRCALPHA)
            step_world = max(8, W//32)
            step_px = max(8, int(s*step_world))
            for x in range(0, world_px[0], step_px):
                pygame.draw.line(grid, (255,255,255,20), (x,0), (x,world_px[1]))
            for y in range(0, world_px[1], step_px):
                pygame.draw.line(grid, (255,255,255,20), (0,y), (world_px[0],y))
            surface.blit(grid, topleft)

    # Ecology balancing (scenario helper, optional)
    balanced_ecology = True
    def maintain_ecology():
        try:
            plant_mask = (world.element_types == 'life') & world.is_alive_mask
            animal_mask = (world.element_types == 'animal') & world.is_alive_mask
            plant_count = int(np.sum(plant_mask))
            animal_count = int(np.sum(animal_mask))
            target_plants = max(20, animal_count * 2)
            deficit = max(0, target_plants - plant_count)
            add_now = min(deficit, 3)
            for _ in range(add_now):
                pid = f'plant_auto_{world.time_step}_{random.randint(0,9999)}'
                world.add_cell(pid, properties={'label': 'bush', 'element_type': 'life', 'position': rand_pos()})
        except Exception:
            pass

    while running:
        dt = clock.tick(args.fps) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            # Handle zoom and pan
            if e.type == pygame.KEYDOWN and e.key == pygame.K_c:
                cinematic_focus = not cinematic_focus
                ui_notify(f"?쒕꽕留덊떛 ?ъ빱?? {'耳쒖쭚' if cinematic_focus else '爰쇱쭚'}")
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                paused = not paused
                ui_notify('?쇱떆?뺤?' if paused else '?ш컻')
            # Tempo control: adjust steps per second (sim_rate)
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                sim_rate = min(16.0, sim_rate * 2.0)
                ui_notify(f"諛곗냽 x{sim_rate:.2f}")
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                sim_rate = max(0.01, sim_rate / 2.0)
                ui_notify(f"諛곗냽 x{sim_rate:.2f}")
            if e.type == pygame.KEYDOWN and e.key == pygame.K_1:
                sim_rate = 0.10; ui_notify("諛곗냽 x0.10")
            if e.type == pygame.KEYDOWN and e.key == pygame.K_2:
                sim_rate = 0.25; ui_notify("諛곗냽 x0.25")
            if e.type == pygame.KEYDOWN and e.key == pygame.K_3:
                sim_rate = 0.50; ui_notify("諛곗냽 x0.50")
            if e.type == pygame.KEYDOWN and e.key == pygame.K_4:
                sim_rate = 1.00; ui_notify("諛곗냽 x1.00")
            if e.type == pygame.KEYDOWN and e.key == pygame.K_5:
                sim_rate = 2.00; ui_notify("諛곗냽 x2.00")
            if e.type == pygame.KEYDOWN and e.key == pygame.K_6:
                sim_rate = 4.00; ui_notify("배속 x4.00")
            if e.type == pygame.KEYDOWN and e.key == pygame.K_7:
                sim_rate = 8.00; ui_notify("배속 x8.00")
            if e.type == pygame.KEYDOWN and e.key == pygame.K_x:
                skill_mode = 'aoe' if skill_mode == 'target' else 'target'
                ui_notify(f"skill mode: {skill_mode}")
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_w, pygame.K_e, pygame.K_r):
                key_map = {pygame.K_q:'Q', pygame.K_w:'W', pygame.K_e:'E', pygame.K_r:'R'}
                sk = key_map.get(e.key)
                if sk:
                    if skill_mode == 'target' and selected_id is not None:
                        idx_sel = world.id_to_idx.get(selected_id)
                        if idx_sel is not None:
                            _apply_skill_to_index(sk, idx_sel)
                            ui_notify(f"{sk} -> {selected_id}")
                    else:
                        mx,my = pygame.mouse.get_pos()
                        wx, wy = screen_to_world(mx, my, scale)
                        n = _apply_skill_aoe(sk, wx, wy, skill_aoe_radius)
                        ui_notify(f"{sk} aoe x{n}")
            if e.type == pygame.KEYDOWN and e.key == pygame.K_g:
                show_grid = not show_grid
                ui_notify(f"洹몃━?? {'耳쒖쭚' if show_grid else '爰쇱쭚'}")
            if e.type == pygame.KEYDOWN and e.key == pygame.K_t:
                show_terrain = not show_terrain
                ui_notify(f"吏?? {'耳쒖쭚' if show_terrain else '爰쇱쭚'}")
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                show_threat = not show_threat
                ui_notify(f"?꾪삊?? {'耳쒖쭚' if show_threat else '爰쇱쭚'}")
            if e.type == pygame.KEYDOWN and e.key == pygame.K_m:
                show_labels = not show_labels
                ui_notify(f"?쇰꺼: {'耳쒖쭚' if show_labels else '爰쇱쭚'}")
            if e.type == pygame.KEYDOWN and e.key == pygame.K_h:
                show_help = not show_help
                ui_notify('?꾩?留??쒖떆' if show_help else '?꾩?留??④?')
            if e.type == pygame.KEYDOWN and e.key == pygame.K_F5:
                try:
                    world_persistence.save_world_state(world)
                    ui_notify('??λ맖 (F5)')
                except Exception as ex:
                    ui_notify(f'저장 실패: {ex}')
            if e.type == pygame.KEYDOWN and e.key == pygame.K_F9:
                try:
                    new_world = world_persistence.load_world_state(world=world, wave_mechanics=mock_wave_mechanics)
                    world = new_world
                    ui_notify('遺덈윭??(F9)')
                except Exception as ex:
                    ui_notify(f'遺덈윭?ㅺ린 ?ㅽ뙣: {ex}')
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                # select nearest alive cell
                mx,my = e.pos
                def dist2_screen(i):
                    sx,sy = w2s(world.positions[i][0], world.positions[i][1])
                    return (sx-mx)**2 + (sy-my)**2
                alive = np.where(world.is_alive_mask)[0]
                if alive.size:
                    idx = min(alive.tolist(), key=dist2_screen)
                    sx,sy = w2s(world.positions[idx][0], world.positions[idx][1])
                    if (sx-mx)**2 + (sy-my)**2 < 20**2:
                        selected_id = world.cell_ids[idx]
                        trail = []
                        ui_notify(f"?좏깮: {selected_id}")
            # Handle zoom and pan
            if e.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                base = min(screen.get_width() / W, screen.get_height() / H)
                s_old = base * scale
                cx_old = (screen.get_width() - int(W * s_old)) // 2
                cy_old = (screen.get_height() - int(H * s_old)) // 2
                # world coords under cursor before zoom
                px_world = (mx - cx_old + pan_x) / max(1e-6, s_old)
                py_world = (my - cy_old + pan_y) / max(1e-6, s_old)
                # apply zoom
                scale = max(0.5, min(8.0, scale * (1.1 if e.y > 0 else 0.9)))
                s_new = base * scale
                cx_new = (screen.get_width() - int(W * s_new)) // 2
                cy_new = (screen.get_height() - int(H * s_new)) // 2
                # keep cursor anchored on same world point
                pan_x = cx_new + px_world * s_new - mx
                pan_y = cy_new + py_world * s_new - my
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 2:
                dragging, last_mouse = True, e.pos
            if e.type == pygame.MOUSEBUTTONUP and e.button == 2:
                dragging = False
            if e.type == pygame.MOUSEMOTION and dragging:
                dx, dy = e.pos[0] - last_mouse[0], e.pos[1] - last_mouse[1]
                pan_x, pan_y = pan_x - dx, pan_y - dy
                last_mouse = e.pos

        # Simulation step(s) paced by real time
        if not paused and sim_rate > 0:
            sim_accum += dt
            interval = 1.0 / sim_rate
            while sim_accum >= interval:
                world.run_simulation_step()
                # Optional ecology maintenance to avoid mass die-off before emergence
                if balanced_ecology and (world.time_step % 5 == 0):
                    maintain_ecology()
                sim_accum -= interval

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
                        event_ticker.append((time.time(), f"{data['actor_id']}媛 {data['target_id']}瑜?癒뱀쓬"))

                elif event_type == 'DEATH' and 'cell_id' in data:
                    dying_cells[data['cell_id']] = time.time()
                    idx = world.id_to_idx.get(data['cell_id'])
                    if idx is not None:
                        pos = world.positions[idx]
                        impact_anims.append(FocusPulse(tuple(pos[:2])))
                    event_ticker.append((time.time(), f"{data['cell_id']} ?щ쭩"))

                elif event_type == 'LIGHTNING_STRIKE' and 'cell_id' in data:
                    idx = world.id_to_idx.get(data['cell_id'])
                    if idx is not None:
                        pos = world.positions[idx]
                        impact_anims.append(LightningBolt(tuple(pos[:2])))
                        impact_anims.append(HitFlash(tuple(pos[:2])))
                        event_ticker.append((time.time(), f"??踰덇컻媛 {data['cell_id']}瑜?媛뺥?"))

        except (IOError, json.JSONDecodeError):
            pass # File might not exist yet or be empty


        # Compute world transform for background rendering
        base = min(screen.get_width() / W, screen.get_height() / H)
        s = max(0.5, min(8.0, scale)) * base
        cx, cy = (screen.get_width() - int(W * s)) // 2, (screen.get_height() - int(H * s)) // 2

        # Background terrain lens (aligned with world transform)
        draw_terrain(screen, s, cx, cy, pan_x, pan_y)
        # Threat heatmap overlay (red tint where high)
        if show_threat and hasattr(world, 'threat_field'):
            tf = world.threat_field
            if tf is not None and tf.size:
                norm = tf.copy()
                if norm.max() > 0:
                    norm = norm / norm.max()
                heat = np.zeros((norm.shape[0], norm.shape[1], 3), dtype=np.uint8)
                heat[...,0] = (norm*255).astype(np.uint8)
                heat[...,1] = (norm*80).astype(np.uint8)
                heat[...,2] = 0
                hs = pygame.surfarray.make_surface(np.rot90(heat))
                world_px = (max(1, int(W*s)), max(1, int(H*s)))
                hs = pygame.transform.smoothscale(hs, world_px)
                hs.set_alpha(120)
                screen.blit(hs, (int(cx - pan_x), int(cy - pan_y)))

        # base/s/cx/cy already computed above for rendering; reuse

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

        # Determine hover candidate for highlighting
        mx, my = pygame.mouse.get_pos()
        hover_idx = None
        alive = np.where(world.is_alive_mask)[0]
        if alive.size:
            def d2(i):
                sx, sy = w2s(world.positions[i][0], world.positions[i][1])
                return (sx - mx)**2 + (sy - my)**2
            hi = min(alive.tolist(), key=d2)
            sxh, syh = w2s(world.positions[hi][0], world.positions[hi][1])
            if (sxh - mx)**2 + (syh - my)**2 < 18**2:
                hover_idx = hi

        # AoE preview ring when in aoe skill mode
        if skill_mode == 'aoe':
            wx, wy = screen_to_world(mx, my, scale)
            rx = int(skill_aoe_radius * s)
            cxp, cyp = w2s(wx, wy)
            pygame.draw.circle(screen, (200, 220, 255), (cxp, cyp), rx, 1)

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

            # scale unit size with zoom for readability
            base_px = min(screen.get_width() / W, screen.get_height() / H)
            zoom_ratio = max(0.5, min(2.5, float((s / max(1e-6, base_px)))))
            size = max(2, min(12, int(4 * zoom_ratio)))
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
            # Body glyphs (simple shapes per species/gender)
            cx_local, cy_local = (size+2, size+2)
            species = world.labels[i] if world.labels.size>i and world.labels[i] else ''
            gender = world.genders[i] if world.genders.size>i else ''
            etype = world.element_types[i] if world.element_types.size>i else ''
            # base
            pygame.draw.circle(temp_surf, color + (alpha,), (cx_local, cy_local), size)
            try:
                if species == 'human' or world.culture[i] in ['wuxia','knight']:
                    if gender == 'male':
                        pts = [(cx_local, cy_local+size+3), (cx_local-6, cy_local-2), (cx_local+6, cy_local-2)]
                        pygame.draw.polygon(temp_surf, (200,200,220,alpha), pts, 0)
                    elif gender == 'female':
                        pts = [(cx_local, cy_local-size-2), (cx_local-6, cy_local), (cx_local, cy_local+6), (cx_local+6, cy_local)]
                        pygame.draw.polygon(temp_surf, (220,180,220,alpha), pts, 0)
                elif species == 'wolf' or (etype=='animal' and species!='deer' and species!='human'):
                    # two ears
                    pygame.draw.polygon(temp_surf, (220,220,220,alpha), [(cx_local-6, cy_local-6),(cx_local-2, cy_local-12),(cx_local+2, cy_local-6)])
                    pygame.draw.polygon(temp_surf, (220,220,220,alpha), [(cx_local+6, cy_local-6),(cx_local+2, cy_local-12),(cx_local-2, cy_local-6)])
                elif etype == 'life':
                    # leaf
                    pygame.draw.ellipse(temp_surf, (80,200,100,alpha), pygame.Rect(cx_local-3, cy_local-10, 6, 14))
                    pygame.draw.line(temp_surf, (60,160,80,alpha), (cx_local, cy_local-10), (cx_local, cy_local+2), 1)
            except Exception:
                pass
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

            # Hover/Selection rings
            if hover_idx == i:
                pygame.draw.circle(screen, (240,240,120), (sx, sy), size+6, 1)
            if selected_id == cell_id:
                pygame.draw.circle(screen, (120,200,255), (sx, sy), size+8, 2)

            # Optional label (name/species)
            if show_labels:
                species_map = {
                    'human':'?щ엺', 'wolf':'?묐?', 'deer':'?ъ뒾', 'tree':'?섎Т', 'life':'?앸챸', 'animal':'?숇Ъ'
                }
                base = species_map.get(species, species_map.get(etype, species or '媛쒖껜'))
                if species == 'human' and gender:
                    base += f"({'남' if gender == 'male' else '여'})"
                label_text = f"{base}"
                label_surf, _ = font.render(label_text, fgcolor=(235,235,245))
                screen.blit(label_surf, (sx - label_surf.get_width()//2, sy + size + 8))

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
        hud_lines = [
            f"?쒓컙 {hh:02d}:{mm:02d}",
            f"媛쒖껜??{int(np.sum(world.is_alive_mask))}",
            f"諛곗냽 x{sim_rate:.2f}{' (?쇱떆?뺤?)' if paused else ''}",
        ]
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

        # Help overlay (top-left)
        if show_help:
            help_lines = [
                '조작 — H로 표시/숨김',
                '마우스: 휠 줌, 가운데 드래그 이동, 좌클릭 선택',
                'ESC 종료  |  Space 일시정지/재개  |  +/- 배속  |  1~7 배속 프리셋',
                '보기: G 그리드, T 지형, M 라벨, C 포커스, R 위협장',
                '레이어: A 에이전트, S 구조물, F 식물, a 동물, W 의지',
                '힌트: 점을 클릭하면 상세 표시, 좌하단: 이벤트',
            ]
            hsurfs = [font.render(l, fgcolor=(235,235,245))[0] for l in help_lines]
            wmax = max(line_surf.get_width() for line_surf in hsurfs) + 14
            hsum = sum(line_surf.get_height() for line_surf in hsurfs) + 14
            panel = pygame.Surface((wmax, hsum), pygame.SRCALPHA)
            panel.fill((0,0,0,150))
            y = 7
            for line_surf in hsurfs:
                panel.blit(line_surf, (7,y)); y += line_surf.get_height()
            screen.blit(panel, (10, 10))

        # Selection detail panel (bottom-right)
        if selected_id is not None:
            idx = world.id_to_idx.get(selected_id)
            if idx is not None and world.is_alive_mask[idx]:
                # trail
                trail.append((float(world.positions[idx][0]), float(world.positions[idx][1])))
                if len(trail) > 50: trail = trail[-50:]
                for j in range(1, len(trail)):
                    x1,y1 = w2s(*trail[j-1]); x2,y2 = w2s(*trail[j])
                    pygame.draw.line(screen, (120,200,255), (x1,y1), (x2,y2), 1)

                # Build selection detail panel
                name = world.labels[idx] if world.labels.size>idx and world.labels[idx] else selected_id
                gender = world.genders[idx] if world.genders.size>idx else ''
                culture = world.culture[idx] if world.culture.size>idx else ''
                cls = f"{culture or 'commoner'}"
                age = int(world.age[idx]) if world.age.size>idx else 0
                max_age = int(world.max_age[idx]) if world.max_age.size>idx else 0
                def talents():
                    ts = []
                    if world.strength[idx] >= 12: ts.append('Brute')
                    if world.agility[idx] >= 12: ts.append('Swift')
                    if world.intelligence[idx] >= 12: ts.append('Sage')
                    if world.wisdom[idx] >= 12: ts.append('Monk')
                    if world.vitality[idx] >= 12: ts.append('Stout')
                    return ', '.join(ts) or '-'

                lines = [
                    f"{name} ({selected_id})",
                    f"Class {cls}  |  Gender {gender or '-'}",
                    f"Age {age}/{max_age}",
                ]
                base_surfs = [font.render(l, fgcolor=(235,235,245))[0] for l in lines]
                # Stat lines
                stat_line = f"STR {world.strength[idx]}  AGI {world.agility[idx]}  INT {world.intelligence[idx]}  VIT {world.vitality[idx]}  WIS {world.wisdom[idx]}"
                talents_line = f"Talents {talents()}"
                stat_surfs = [font.render(stat_line, fgcolor=(220,230,240))[0], font.render(talents_line, fgcolor=(220,230,240))[0]]

                # Build panel size
                wmax = max([surf_l.get_width() for surf_l in base_surfs + stat_surfs] + [200]) + 16
                hsum = sum(surf_l.get_height() for surf_l in base_surfs + stat_surfs) + 8 + 4*8 + 20
                panel = pygame.Surface((wmax, hsum), pygame.SRCALPHA)
                panel.fill((0,0,0,150))
                y = 6
                for surf_l in base_surfs:                    panel.blit(surf_l, (8, y)); y += surf_l.get_height()

                # Resource bars HP/Ki/Mana/Faith
                def draw_bar(label, curr, maxv, color):
                    nonlocal y
                    bar_w, bar_h = wmax - 16 - 70, 6
                    x0 = 8 + 70
                    ratio = 0.0 if maxv <= 0 else max(0.0, min(1.0, float(curr/maxv)))
                    txt, _ = font.render(f"{label}", fgcolor=(235,235,245))
                    panel.blit(txt, (8, y-1))
                    pygame.draw.rect(panel, (50,60,70,200), pygame.Rect(x0, y, bar_w, bar_h), border_radius=2)
                    pygame.draw.rect(panel, color, pygame.Rect(x0, y, int(bar_w*ratio), bar_h), border_radius=2)
                    y += bar_h + 6

                draw_bar('HP', world.hp[idx], world.max_hp[idx], (80,220,120,230))
                draw_bar('Ki', world.ki[idx] if world.ki.size>idx else 0, world.max_ki[idx] if world.max_ki.size>idx else 0, (120,200,255,230))
                draw_bar('MP', world.mana[idx] if world.mana.size>idx else 0, world.max_mana[idx] if world.max_mana.size>idx else 0, (120,120,255,230))
                draw_bar('Faith', world.faith[idx] if world.faith.size>idx else 0, world.max_faith[idx] if world.max_faith.size>idx else 0, (240,200,120,230))

                # Stat lines
                for surf_l in stat_surfs:                    panel.blit(surf_l, (8, y)); y += surf_l.get_height()

                screen.blit(panel, (screen.get_width()-wmax-10, screen.get_height()-hsum-10))


        pygame.display.flip()

    pygame.quit(); sys.exit()


if __name__ == '__main__':
    main()




