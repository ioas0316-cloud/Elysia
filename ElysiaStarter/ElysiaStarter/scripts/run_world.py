import argparse, time, os
from core.cell_world import CellWorld
from core.biome import classify_biome
from core.agents import Agents
import yaml
import numpy as np

def _load_cfg():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(here, 'config', 'runtime.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run(steps=1000, render=False, dt=1.0):
    cfg = _load_cfg()
    W, H = cfg.get('world', {}).get('grid', [256, 256])
    n_agents = cfg.get('agents', {}).get('count', 8000)

    qoe = cfg.get('qoe', {})
    qoe_on = (qoe.get('mode', 'off') == 'on')
    tile_w, tile_h = qoe.get('tile', [32, 32])
    radius = int(qoe.get('focus_radius_tiles', 3))
    budget = int(qoe.get('budget_tiles_per_step', 24))
    off_int = int(qoe.get('offscreen_interval', 30))
    freeze_outside = bool(qoe.get('freeze_agents_outside', True))
    freeze_rate = float(qoe.get('freeze_rate', 0.0))

    world = CellWorld(W,H)
    agents = Agents(n_agents, W,H)
    last_print = time.time()
    biome = None
    # tile partition
    tiles_x = max(1, W // tile_w)
    tiles_y = max(1, H // tile_h)
    def agent_tiles(ax, ay):
        tx = np.clip((ax // tile_w).astype(int), 0, tiles_x-1)
        ty = np.clip((ay // tile_h).astype(int), 0, tiles_y-1)
        return tx, ty

    focus = (W//2, H//2)

    for i in range(steps):
        # World fields (simple: full update for now; 256x256 is light on CPU)
        world.update_fields()
        biome = classify_biome(world.height, world.moisture, world.temp)

        # QOE scheduler: compute visible tiles around focus, cap by budget
        tx_focus = focus[0] // tile_w
        ty_focus = focus[1] // tile_h
        visible = []
        for ty in range(max(0, ty_focus-radius), min(tiles_y, ty_focus+radius+1)):
            for tx in range(max(0, tx_focus-radius), min(tiles_x, tx_focus+radius+1)):
                visible.append((tx,ty))
        # cap budget
        visible = visible[:budget]
        visible_set = set(visible)

        # Per-agent rate based on whether agent is in visible tiles
        prev_x = agents.x.copy(); prev_y = agents.y.copy()
        prev_hunger = agents.hunger.copy(); prev_energy = agents.energy.copy()

        if qoe_on and freeze_outside:
            atx, aty = agent_tiles(agents.x, agents.y)
            mask_visible = np.array([(tx,ty) in visible_set for tx,ty in zip(atx,aty)])
            rate = np.where(mask_visible, 1.0, freeze_rate).astype(np.float32)
        else:
            rate = None

        agents.step(biome, dt=dt)

        # If QOE freeze is on, blend or restore outside agents
        if qoe_on and freeze_outside:
            if freeze_rate == 0.0:
                # restore outside
                outside = (rate < 1.0)
                agents.x[outside] = prev_x[outside]
                agents.y[outside] = prev_y[outside]
                agents.hunger[outside] = prev_hunger[outside]
                agents.energy[outside] = prev_energy[outside]
            else:
                # interpolate by rate for outside agents
                outside = (rate < 1.0)
                r = rate[outside]
                agents.x[outside] = prev_x[outside] + r*(agents.x[outside]-prev_x[outside])
                agents.y[outside] = prev_y[outside] + r*(agents.y[outside]-prev_y[outside])
                agents.hunger[outside] = prev_hunger[outside] + r*(agents.hunger[outside]-prev_hunger[outside])
                agents.energy[outside] = prev_energy[outside] + r*(agents.energy[outside]-prev_energy[outside])

        if time.time()-last_print>1.0:
            tiles_full = len(visible)
            tiles_lod = tiles_x*tiles_y - tiles_full
            print(f"[step {i}] QOE: tiles(full)={tiles_full} tiles(LOD)={tiles_lod} agents={agents.N} dt={dt}")
            last_print=time.time()
    if render:
        try:
            import pygame, numpy as np
            pygame.init()
            screen=pygame.display.set_mode((512,512))
            running=True
            while running:
                for e in pygame.event.get(): 
                    if e.type==pygame.QUIT: running=False
                import numpy as np
                surf = pygame.surfarray.make_surface((biome/biome.max()*255).astype('uint8').T.repeat(2,0).repeat(2,1))
                screen.blit(surf,(0,0))
                pygame.display.flip()
            pygame.quit()
        except Exception as e:
            print("Render skipped:", e)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--render", type=int, default=0)
    ap.add_argument("--dt", type=float, default=1.0, help="simulation time delta per step (time acceleration)")
    args = ap.parse_args()
    run(steps=args.steps, render=bool(args.render), dt=args.dt)
