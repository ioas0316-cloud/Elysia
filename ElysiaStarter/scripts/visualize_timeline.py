import argparse
import os
import sys
import time

import numpy as np
import pygame
import yaml


# Ensure Starter package root is on sys.path (ElysiaStarter)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_THIS_DIR)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from core.cell_world import CellWorld
from core.biome import classify_biome
from ui.view_state import load_view_state, save_view_state
from ui.layer_panel import handle_layer_keys, draw_layer_hud, draw_layer_panel
from ui.fonts import get_font


def load_cfg():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(here, 'config', 'runtime.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def draw_help(screen: pygame.Surface, font: "pygame.freetype.Font", y_start: int) -> int:
    lines = [
        '도움말 (H)',
        '이동: 방향키 / WASD',
        '줌: 마우스 휠 (커서 고정)',
        '이동(팬): 마우스 중클릭 드래그',
        '종료: Q',
    ]
    pad = 10
    line_h = max(18, font.get_height() + 2)
    h = pad + len(lines) * line_h + pad
    surfs = [font.render(t, fgcolor=(220, 220, 230))[0] for t in lines]
    w = max(240, max(s.get_width() for s in surfs) + 20)
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


def biome_to_rgb(biome: np.ndarray) -> np.ndarray:
    # Simple palette for demo: 0..N -> RGB
    colors = np.array([
        [30, 30, 40],    # 0
        [60, 120, 200],  # 1 water
        [180, 180, 140], # 2 sand
        [110, 170, 80],  # 3 grass
        [40, 110, 50],   # 4 forest
        [140, 120, 120], # 5 rock
    ], dtype=np.uint8)
    idx = np.clip(biome.astype(np.int32), 0, len(colors) - 1)
    rgb = colors[idx]
    return rgb


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
    scale = 1.0  # 0.5~4.0 배율
    pan_x = 0.0
    pan_y = 0.0
    dragging = False
    last_mouse = (0, 0)
    last_print = time.time()
    help_y = 10

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            handle_layer_keys(e)
            if e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                running = False
            # 줌: 마우스 휠 (커서 고정)
            if e.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                base = min(screen.get_width() / W, screen.get_height() / H)
                s_old = base * scale
                scale = max(0.5, min(4.0, scale * (1.1 if e.y > 0 else 0.9)))
                s_new = base * scale
                # 커서 아래 세계 좌표 고정: pan 보정
                u = mx + pan_x
                v = my + pan_y
                pan_x = (u / s_old) * s_new - mx
                pan_y = (v / s_old) * s_new - my
            # 팬: 중클릭 드래그
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

        # Update world and classify
        world.update_fields()
        biome = classify_biome(world.height, world.moisture, world.temp)

        # Render biome as image with 논리 줌/팬
        rgb = biome_to_rgb(biome)  # (H,W,3)
        base_surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))  # (W,H)
        base = min(screen.get_width() / W, screen.get_height() / H)
        s = max(0.5, min(4.0, scale)) * base
        scaled_w = max(1, int(W * s))
        scaled_h = max(1, int(H * s))
        surf = pygame.transform.smoothscale(base_surf, (scaled_w, scaled_h))
        # 화면 중앙 기준 정렬 + 팬 적용
        cx = (screen.get_width() - scaled_w) // 2
        cy = (screen.get_height() - scaled_h) // 2
        screen.blit(surf, (cx - pan_x, cy - pan_y))

        # HUD: simple help + layer indicators
        draw_layer_hud(screen)
        draw_layer_panel(screen)
        help_y = draw_help(screen, font, help_y)

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
