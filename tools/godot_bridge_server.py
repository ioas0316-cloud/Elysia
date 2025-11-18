"""
Godot Bridge Server (WebSocket)

Purpose
- Expose the Python World (Project_Sophia.core.world.World) to a Godot 4 client
  via a simple WebSocket protocol.
- Keep simulation authoritative in Python; Godot focuses on rendering & input.

Protocol (JSON frames)
- init: sent once on connect
    {
      "type": "init",
      "world": {"width": int},
      "lenses": ["threat","value","will"],
      "tick": int
    }
- frame: sent periodically (every n simulation steps)
    {
      "type": "frame",
      "tick": int,
      "cells": [{"id": str, "x": float, "y": float, "type": str, "alive": bool}],
      "overlays": {"threat": str(b64-png), "value": str(b64-png), "will": str(b64-png)}
    }
- input: received from client
    {
      "type": "input",
      "sim_rate": float(optional),
      "lens": {"threat": bool, "value": bool, "will": bool}(optional),
      "select_id": str(optional),
      "disaster": {"kind": "FLOOD"|"VOLCANO", "x": int, "y": int, "radius": int}(optional)
    }

Run
    python tools/godot_bridge_server.py --host 127.0.0.1 --port 8765 --rate 4 --frame-every 1

Requirements
- websockets, pillow, numpy (already in requirements.txt)

Note
- World is square: width x width (height == width)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import signal
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

# Ensure repo root on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
import sys
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# World imports
from Project_Sophia.core.world import World  # type: ignore
from Project_Sophia.wave_mechanics import WaveMechanics  # type: ignore
from tools.kg_manager import KGManager  # type: ignore
from Project_Sophia.world_themes.west_continent.characters import (  # type: ignore
    WEST_CHARACTER_POOL,
)

import websockets
from websockets.server import WebSocketServerProtocol


@dataclass
class BridgeConfig:
    host: str = "127.0.0.1"
    # Use a high, rarely-used default port to avoid conflicts.
    port: int = 8877
    sim_rate: float = 4.0  # steps per second
    frame_every: int = 1  # send a frame every N simulation steps
    max_cells: int = 5000  # cap cells included per frame


class GodotBridge:
    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        # Build a simple world like the visualizer
        kg = KGManager()
        wm = WaveMechanics(kg)
        self.world = World(primordial_dna={'instinct': 'observe'}, wave_mechanics=wm)
        # Seed a few entities for first view
        self._seed_world()
        self._clients: List[WebSocketServerProtocol] = []
        self._running = True
        self._sim_last = time.time()
        self._accum = 0.0

    def _seed_world(self) -> None:
        """
        Seed the world with a simple West Continent themed population and terrain.

        Goal: when WorldView connects, it should immediately see a living map
        (terrain, river, farms, threat/value gradients) and a dense scatter
        of human cells (~2000) instead of an empty grid.
        """
        rng = np.random.default_rng(42)
        W = int(getattr(self.world, "width", 256))

        def rand_pos() -> Dict[str, float]:
            return {
                "x": float(rng.uniform(0, W - 1)),
                "y": float(rng.uniform(0, W - 1)),
                "z": 0.0,
            }

        # --- Human population (~800 instances from the West Continent pool) ---
        pool = list(WEST_CHARACTER_POOL)
        pool_size = max(1, len(pool))
        # 800 is enough to make the map feel alive, while keeping startup cost low.
        target_count = 800

        # Pre-expand world matrices once to avoid repeated costly resizes
        try:
            adj = getattr(self.world, "adjacency_matrix", None)
            current_size = int(adj.shape[0]) if adj is not None else 0
            desired = max(target_count + 64, current_size)
            if hasattr(self.world, "_resize_matrices") and desired > current_size:
                # Pre-expand once so add_cell does not trigger many resizes.
                self.world._resize_matrices(desired)  # type: ignore[attr-defined]
        except Exception:
            # If pre-resize fails, seeding still proceeds; it may just be slower.
            pass
        for i in range(target_count):
            tmpl = pool[i % pool_size]
            cid = f"{tmpl.id}_h{i:04d}"
            props = {
                "label": tmpl.role,
                "element_type": tmpl.element_type,
                "culture": tmpl.culture,
                "display_name": tmpl.display_name,
                "position": rand_pos(),
            }
            props.update(tmpl.notes)
            try:
                self.world.add_cell(cid, properties=props)
            except Exception:
                # Seeding should never crash the bridge; skip bad entries.
                continue

        # --- A few non-human markers (trees as 'life') ---
        for i in range(80):
            cid = f"tree_{i:03d}"
            try:
                self.world.add_cell(
                    cid,
                    properties={
                        "label": "tree",
                        "element_type": "life",
                        "position": rand_pos(),
                    },
                )
            except Exception:
                continue

        # --- Terrain fields for overlays ---
        try:
            # Value mass: gentle east-west gradient (warmer colors on the right)
            x = np.linspace(0.0, 1.0, W, dtype=np.float32)
            vm = np.tile(x, (W, 1))
            if hasattr(self.world, "value_mass_field"):
                self.world.value_mass_field = vm

            # Threat: stronger toward the north (top of the map)
            y = np.linspace(1.0, 0.0, W, dtype=np.float32)
            threat = np.tile(y[:, None], (1, W))
            if hasattr(self.world, "threat_field"):
                self.world.threat_field = threat

            # Wetness: create a simple river that snakes horizontally.
            wet = np.zeros((W, W), dtype=np.float32)
            for x_idx in range(W):
                center_y = int(W * 0.35 + 8.0 * np.sin(x_idx / 18.0))
                y0 = max(0, center_y - 2)
                y1 = min(W, center_y + 3)
                wet[y0:y1, x_idx] = 0.85
            if hasattr(self.world, "wetness"):
                self.world.wetness = wet
        except Exception:
            # If any of this fails, we still have a populated world; overlays just stay minimal.
            pass

        try:
            print(f"[Bridge] Seeded world with ~{target_count} humans and terrain fields.")
        except Exception:
            pass

    async def start(self) -> None:
        async with websockets.serve(self._client_handler, self.cfg.host, self.cfg.port, ping_interval=20, ping_timeout=20):
            print(f"[Bridge] Listening on ws://{self.cfg.host}:{self.cfg.port}")
            # simulation loop
            try:
                while self._running:
                    now = time.time()
                    dt = now - self._sim_last
                    self._sim_last = now
                    self._accum += dt
                    interval = 1.0 / max(1e-3, self.cfg.sim_rate)
                    sent = False
                    while self._accum >= interval:
                        self.world.run_simulation_step()
                        self._accum -= interval
                        if (self.world.time_step % max(1, self.cfg.frame_every)) == 0:
                            await self._broadcast_frame()
                            sent = True
                    # If no step fit (very low rate), still drip frames sometimes
                    if not sent and self.world.time_step % max(1, self.cfg.frame_every) == 0:
                        await self._broadcast_frame()
                    await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                pass
            finally:
                print("[Bridge] Stopped")

    async def _client_handler(self, ws: WebSocketServerProtocol) -> None:
        self._clients.append(ws)
        try:
            await ws.send(json.dumps(self._build_init()))
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                await self._handle_input(data)
        finally:
            if ws in self._clients:
                self._clients.remove(ws)

    def _build_init(self) -> Dict[str, Any]:
        return {
            'type': 'init',
            'world': {'width': int(getattr(self.world, 'width', 256))},
            'lenses': ['threat', 'value', 'will', 'coherence'],
            'tick': int(self.world.time_step),
        }

    async def _broadcast_frame(self) -> None:
        """Send a frame to all connected clients, pruning closed sockets."""
        if not self._clients:
            return
        payload = self._build_frame()
        js = json.dumps(payload)
        # Send defensively: drop clients that closed during broadcast
        for c in list(self._clients):
            try:
                # Some versions expose .closed flag/state
                if getattr(c, "closed", False):
                    self._clients.remove(c)
                    continue
                await c.send(js)
            except Exception:
                # On any send error, close and forget the client
                try:
                    await c.close()
                except Exception:
                    pass
                if c in self._clients:
                    self._clients.remove(c)

    def _build_frame(self) -> Dict[str, Any]:
        w = int(getattr(self.world, "width", 256))
        # cells slice
        cells: List[Dict[str, Any]] = []
        if getattr(self.world, "cell_ids", None):
            alive_mask = (
                self.world.is_alive_mask
                if getattr(self.world, "is_alive_mask", None) is not None
                else np.ones((len(self.world.cell_ids),), dtype=bool)
            )
            count = min(len(self.world.cell_ids), self.cfg.max_cells)
            for i in range(count):
                if i >= self.world.positions.shape[0]:
                    break
                cid = self.world.cell_ids[i]
                pos = self.world.positions[i]
                alive = bool(alive_mask[i])
                label = ""
                try:
                    label = self.world.element_types[i]
                except Exception:
                    label = ""
                cells.append(
                    {
                        "id": cid,
                        "x": float(pos[0]),
                        "y": float(pos[1]),
                        "type": label,
                        "alive": alive,
                    }
                )

        # Debug: print once in a while how many cells are present.
        try:
            if self.world.time_step % 60 == 0:
                print(f"[Bridge] frame at tick={self.world.time_step}, cells={len(cells)}")
        except Exception:
            pass

        overlays = {
            "terrain": self._encode_terrain_rgb(),
            "river": self._encode_overlay(self._river_flow()),
            "veg": self._encode_overlay(self._plant_density()),
            "farm": self._encode_overlay(self._farmland_intensity()),
            "farm_paddy": self._encode_overlay(self._farmland_paddy()),
            "farm_field": self._encode_overlay(self._farmland_field()),
            "threat": self._encode_overlay(getattr(self.world, "threat_field", None)),
            "value": self._encode_overlay(getattr(self.world, "value_mass_field", None)),
            "will": self._encode_overlay(getattr(self.world, "will_field", None)),
            "coherence": self._encode_overlay(self._coherence_map()),
        }
        civ = self._build_civ_overlay(w)
        elysia_state = self._read_latest_soul_state()
        return {
            "type": "frame",
            "tick": int(self.world.time_step),
            "cells": cells,
            "overlays": overlays,
            "civ": civ,
            "world": {"width": w},
            "time": {
                "phase": (
                    int(self.world.time_step) % int(getattr(self.world, "day_length", 1) or 1)
                )
                / float(max(1, int(getattr(self.world, "day_length", 1) or 1)))
            },
            "elysia": elysia_state,
        }

    def _read_latest_soul_state(self) -> Dict[str, Any]:
        """
        Best-effort helper to read the latest SoulState snapshot.

        Expected source: elysia_logs/soul_state.jsonl written by the cognition pipeline.
        Returns a small dict with at least {"mood": str} when available.
        """
        try:
            log_dir = os.path.join(_ROOT, "elysia_logs")
            path = os.path.join(log_dir, "soul_state.jsonl")
            if not os.path.exists(path):
                return {}
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            if not lines:
                return {}
            last = lines[-1].strip()
            if not last:
                return {}
            data = json.loads(last)
            if isinstance(data, dict):
                # Keep it light: mood + a few fields only.
                mood = str(data.get("mood", "neutral"))
                return {
                    "mood": mood,
                }
        except Exception:
            pass
        return {}

    def _build_civ_overlay(self, w: int) -> Dict[str, Any]:
        """Build a minimal high-level civ/caravan snapshot.
        For now this is a virtual/dummy layer that can later be wired to real CivNode/Party data.
        """
        mid_y = w * 0.5
        left_x = w * 0.25
        right_x = w * 0.75
        nodes = [
            {'id': 'village_h_1', 'label': 'human_village', 'x': left_x, 'y': mid_y, 'pop': 120, 'wealth': 0.4},
            {'id': 'fae_village_1', 'label': 'fae_village', 'x': right_x, 'y': mid_y, 'pop': 80, 'wealth': 0.3},
        ]
        # simple virtual caravans moving along the line, driven by time_step
        t_base = (self.world.time_step % 240) / 240.0
        caravans = [
            {'id': 'caravan_1', 'origin_id': 'village_h_1', 'target_id': 'fae_village_1', 't': t_base, 'kind': 'trade'},
            {'id': 'caravan_2', 'origin_id': 'fae_village_1', 'target_id': 'village_h_1', 't': (t_base + 0.35) % 1.0, 'kind': 'trade'},
        ]
        return {'nodes': nodes, 'caravans': caravans}

    def _coherence_map(self) -> Optional[np.ndarray]:
        """Compute a lightweight coherence map from value_mass and will field gradients.
        Returns float32 array 0..1 where higher means stronger aligned structure.
        """
        try:
            vm = getattr(self.world, 'value_mass_field', None)
            wl = getattr(self.world, 'will_field', None)
            if vm is None or wl is None:
                return None
            a = np.asarray(vm, dtype=np.float32)
            b = np.asarray(wl, dtype=np.float32)
            if a.size == 0 or b.size == 0 or a.shape != b.shape:
                return None
            # Gradients (y, x)
            gy_a, gx_a = np.gradient(a)
            gy_b, gx_b = np.gradient(b)
            # Magnitudes and dot alignment
            mag_a = np.hypot(gx_a, gy_a)
            mag_b = np.hypot(gx_b, gy_b)
            denom = (mag_a * mag_b) + 1e-6
            dot = (gx_a * gx_b + gy_a * gy_b) / denom  # -1..1
            align = (dot + 1.0) * 0.5  # 0..1
            # Emphasize where both magnitudes are present
            wmag = mag_a * mag_b
            mmax = float(wmag.max()) if wmag.size else 0.0
            if mmax > 0.0:
                wmag = wmag / mmax
            coh = (align * wmag).astype(np.float32)
            return coh
        except Exception:
            return None

    def _encode_overlay(self, arr: Optional[np.ndarray]) -> Optional[str]:
        if arr is None:
            return None
        try:
            a = np.asarray(arr)
            if a.size == 0:
                return None
            # Normalize to 0..255 uint8
            m = float(a.max())
            if m <= 0:
                img8 = np.zeros_like(a, dtype=np.uint8)
            else:
                img8 = (np.clip(a / m, 0.0, 1.0) * 255.0).astype(np.uint8)
            img = Image.fromarray(img8, mode='L')
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode('ascii')
        except Exception:
            return None

    def _encode_color(self, rgb: Optional[np.ndarray]) -> Optional[str]:
        if rgb is None:
            return None
        try:
            arr = np.asarray(rgb, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[2] != 3:
                return None
            img = Image.fromarray(arr, mode='RGB')
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode('ascii')
        except Exception:
            return None

    def _encode_terrain_rgb(self) -> Optional[str]:
        """Build a simple color terrain from height_map/wetness/soil_fertility.
        Colors: water (deep/shallow), sand (shore), grass (plains), rock (mountain).
        """
        try:
            w = int(getattr(self.world, 'width', 256))
            if w <= 0:
                return None
            height = getattr(self.world, 'height_map', None)
            wet = getattr(self.world, 'wetness', None)
            if height is None:
                height = np.zeros((w, w), dtype=np.float32)
            else:
                height = np.asarray(height, dtype=np.float32)
            if wet is None:
                wet = np.zeros((w, w), dtype=np.float32)
            else:
                wet = np.asarray(wet, dtype=np.float32)
            # normalize height
            h = height.copy()
            h -= float(h.min()) if h.size else 0.0
            mx = float(h.max()) if h.size else 1.0
            if mx <= 1e-6:
                mx = 1.0
            h /= mx
            water = wet > 0.6
            shallow = (wet > 0.4) & (wet <= 0.6)
            # mountain threshold
            mountain = (h >= 0.70) & (~water) & (~shallow)
            # shore: neighbor of water
            neigh = np.zeros_like(water, dtype=bool)
            neigh[:-1,:] |= water[1:,:]
            neigh[1: ,:] |= water[:-1,:]
            neigh[:, :-1] |= water[:,1:]
            neigh[:, 1: ] |= water[:,:-1]
            shore = neigh & (~water)
            # base = grass
            rgb = np.zeros((w, w, 3), dtype=np.uint8)
            rgb[...,0] = 60; rgb[...,1] = 110; rgb[...,2] = 70  # grass
            # water/shallow
            rgb[water] = (30, 60, 200)
            rgb[shallow] = (60, 100, 200)
            # shore sand
            rgb[shore] = (170, 150, 110)
            # mountain rock
            rgb[mountain] = (100, 100, 100)
            return self._encode_color(rgb)
        except Exception:
            return None

    def _plant_density(self) -> Optional[np.ndarray]:
        try:
            w = int(getattr(self.world, 'width', 256))
            if w <= 0 or not getattr(self.world, 'cell_ids', None):
                return None
            density = np.zeros((w, w), dtype=np.float32)
            labels = getattr(self.world, 'element_types', None)
            if labels is None:
                return None
            for i, cid in enumerate(self.world.cell_ids):
                if i >= self.world.positions.shape[0]:
                    break
                try:
                    if labels.size > i and labels[i] == 'life':
                        x = int(self.world.positions[i][0]) % w
                        y = int(self.world.positions[i][1]) % w
                        density[y, x] += 1.0
                except Exception:
                    continue
            # simple blur to spread clusters
            for _ in range(3):
                density = (density + np.roll(density,1,0) + np.roll(density,-1,0) + np.roll(density,1,1) + np.roll(density,-1,1)) / 5.0
            return density
        except Exception:
            return None

    
    def _farmland_paddy(self) -> Optional[np.ndarray]:
        try:
            base = self._farmland_intensity()
            if base is None:
                return None
            wet = getattr(self.world, 'wetness', None)
            if wet is None:
                return None
            w = np.asarray(wet, dtype=np.float32)
            mask = (w >= 0.35) & (w <= 0.75)
            out = np.where(mask, base, 0.0).astype(np.float32)
            return out
        except Exception:
            return None

    def _farmland_field(self) -> Optional[np.ndarray]:
        try:
            base = self._farmland_intensity()
            if base is None:
                return None
            wet = getattr(self.world, 'wetness', None)
            w = np.asarray(wet, dtype=np.float32) if wet is not None else None
            if w is None:
                return base
            mask = (w < 0.35)
            out = np.where(mask, base, 0.0).astype(np.float32)
            return out
        except Exception:
            return None

    def _river_flow(self) -> Optional[np.ndarray]:
        """Derive a simple river intensity from wetness with light thinning."""
        try:
            wet = getattr(self.world, 'wetness', None)
            if wet is None:
                return None
            w = np.asarray(wet, dtype=np.float32)
            if w.size == 0:
                return None
            base = np.clip((w - 0.45) / 0.55, 0.0, 1.0)
            river = base > 0.0
            nb = np.zeros_like(base, dtype=np.int16)
            nb[:-1,:] += river[1:,:]
            nb[1: ,:] += river[:-1,:]
            nb[:, :-1] += river[:,1:]
            nb[:, 1: ] += river[:,:-1]
            thin = np.where((river) & (nb <= 2), base, base*0.5)
            return thin.astype(np.float32)
        except Exception:
            return None

    def _farmland_intensity(self) -> Optional[np.ndarray]:
        try:
            w = int(getattr(self.world, 'width', 256))
            if w <= 0:
                return None
            human_d = np.zeros((w, w), dtype=np.float32)
            labels = getattr(self.world, 'element_types', None)
            wet = getattr(self.world, 'wetness', None)
            fert = getattr(self.world, 'soil_fertility', None)
            if labels is None or fert is None:
                return None
            fert = np.asarray(fert, dtype=np.float32)
            wet = np.asarray(wet, dtype=np.float32) if wet is not None else np.zeros((w,w), dtype=np.float32)
            for i, cid in enumerate(self.world.cell_ids):
                if i >= self.world.positions.shape[0]:
                    break
                try:
                    if labels.size > i and labels[i] == 'human':
                        x = int(self.world.positions[i][0]) % w
                        y = int(self.world.positions[i][1]) % w
                        human_d[y, x] += 1.0
                except Exception:
                    continue
            for _ in range(3):
                human_d = (human_d + np.roll(human_d,1,0) + np.roll(human_d,-1,0) + np.roll(human_d,1,1) + np.roll(human_d,-1,1)) / 5.0
            # farmland prefers fertile, near humans, not deep water
            water = wet > 0.5
            inten = (fert * 0.7) + (human_d / (human_d.max()+1e-6)) * 0.3
            inten[water] = 0.0
            return np.clip(inten, 0.0, 1.0)
        except Exception:
            return None

    async def _handle_input(self, data: Dict[str, Any]) -> None:
        if data.get('type') != 'input':
            return
        if 'sim_rate' in data:
            try:
                sr = float(data['sim_rate'])
                self.cfg.sim_rate = max(0.01, min(32.0, sr))
            except Exception:
                pass
        if 'disaster' in data and isinstance(data['disaster'], dict):
            self._apply_disaster(data['disaster'])

    def _apply_disaster(self, d: Dict[str, Any]) -> None:
        kind = (d.get('kind') or '').upper()
        x = int(d.get('x', 0)); y = int(d.get('y', 0)); r = int(d.get('radius', 6))
        W = int(getattr(self.world, 'width', 256))
        x = max(0, min(W-1, x)); y = max(0, min(W-1, y)); r = max(1, min(W//4, r))
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        y0, y1 = max(0, y - r), min(W, y + r + 1)
        if kind == 'FLOOD':
            # increase wetness locally
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    dx = xx - x; dy = yy - y
                    if dx*dx + dy*dy <= r*r:
                        try:
                            self.world.wetness[yy, xx] = min(1.0, self.world.wetness[yy, xx] + 0.7)
                        except Exception:
                            pass
            self.world.event_logger.log('FLOOD', self.world.time_step, x=x, y=y, radius=r)
        elif kind == 'VOLCANO':
            # heat up area (prestige as proxy), reduce wetness
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    dx = xx - x; dy = yy - y
                    if dx*dx + dy*dy <= r*r:
                        try:
                            self.world.prestige_field[yy, xx] = min(1.0, self.world.prestige_field[yy, xx] + 0.5)
                            self.world.wetness[yy, xx] = max(0.0, self.world.wetness[yy, xx] - 0.5)
                        except Exception:
                            pass
            self.world.event_logger.log('VOLCANO', self.world.time_step, x=x, y=y, radius=r)


async def amain(cfg: BridgeConfig) -> None:
    bridge = GodotBridge(cfg)
    stop = asyncio.get_event_loop().create_future()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_event_loop().add_signal_handler(s, lambda: stop.set_result(True))
        except NotImplementedError:
            # Windows may not support signal handlers in event loop
            pass
    await asyncio.gather(bridge.start(), stop)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    # Match BridgeConfig default: use 8877 to avoid common conflicts.
    ap.add_argument("--port", type=int, default=8877)
    ap.add_argument("--rate", type=float, default=4.0, help="simulation steps per second")
    ap.add_argument("--frame-every", type=int, default=1, help="send a frame every N steps")
    ap.add_argument("--max-cells", type=int, default=5000)
    args = ap.parse_args()
    cfg = BridgeConfig(host=args.host, port=args.port, sim_rate=args.rate, frame_every=args.frame_every, max_cells=args.max_cells)
    asyncio.run(amain(cfg))


if __name__ == '__main__':
    main()

