"""
causal_exploration_game.py

Blind Visual-Physical Exploration Game for ELYSIA (L3 Interface)
=================================================================
Provides an interactive, visually rendered, multi-physics 5x5 exploration game built
over genuine 4K Yosemite natural rock/forest textures.

Strict Agent-World Isolation:
- The agent receives ONLY raw uint8 RGB images (`observe_frame`, `foveate_tile`),
  raw 1D force-displacement contact traces (`probe_push_tile`), 1D PCM acoustic taps
  (`probe_tap_tile`), and action outcome scalars (`move`: moved, blocked, repelled,
  energy_delta, level_cleared).
- Hidden tile types, grain angles, stiffnesses, and hazard labels remain strictly
  inside `_GameTile` and are NEVER exposed to any learner method.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from simulators.hidden_multiphysics_world import YOSEMITE_4K_PATH


@dataclass
class _GameTile:
    row: int
    col: int
    kind: str  # Hidden ground truth ONLY
    passable: bool
    is_hazard: bool
    is_goal: bool
    patch_rgb: np.ndarray
    stiffness_n_m: float
    k_perp_n_m: float
    fy_par_n: float
    fy_perp_n: float
    damping_ratio: float
    plastic_softening: float
    peak_freq_hz: float
    decay_rate_s: float
    quality_q: float
    phi_mech_deg: Optional[float] = None
    is_cleavable: bool = False
    is_compressible: bool = False
    cleaved: bool = False


@dataclass
class GameStepOutcome:
    direction: str
    new_pos: Tuple[int, int]
    moved: bool
    blocked: bool
    repelled: bool
    energy_delta: float
    energy_remaining: float
    level_cleared: bool
    frame_rgb: np.ndarray


def _scale_linear_irradiance(patch_u8: np.ndarray, factor: float) -> np.ndarray:
    lin = np.power(patch_u8.astype(np.float32) / 255.0, 2.2)
    lin = np.clip(lin * float(factor), 0.0, 1.0)
    return np.clip(np.round(np.power(lin, 1.0 / 2.2) * 255.0), 0, 255).astype(np.uint8)


def _apply_oriented_grain(
    patch_u8: np.ndarray, angle_deg: float, wavelength_px: float = 6.0, amp: float = 0.36
) -> np.ndarray:
    """Apply sinusoidal striation along angle_deg (0 deg=+x, 90 deg=+y) in linear light."""
    h, w = patch_u8.shape[:2]
    phi = np.deg2rad(float(angle_deg))
    yy, xx = np.indices((h, w), dtype=np.float32)
    s_perp = -xx * float(np.sin(phi)) + yy * float(np.cos(phi))
    mod = (1.0 + float(amp) * np.cos((2.0 * np.pi / float(wavelength_px)) * s_perp)).astype(np.float32)
    lin = np.power(patch_u8.astype(np.float32) / 255.0, 2.2)
    lin = np.clip(lin * mod[..., None], 0.0, 1.0)
    return np.clip(np.round(np.power(lin, 1.0 / 2.2) * 255.0), 0, 255).astype(np.uint8)


def _apply_micropores(patch_u8: np.ndarray, amp: float = 0.34) -> np.ndarray:
    """Apply 4px-period 2x2 high-frequency micropore pattern visible in 1:1 foveation."""
    h, w = patch_u8.shape[:2]
    yy, xx = np.indices((h, w))
    sign = 1.0 - 2.0 * (((yy // 2) + (xx // 2)) % 2)
    mod = (1.0 + float(amp) * sign).astype(np.float32)
    lin = np.power(patch_u8.astype(np.float32) / 255.0, 2.2)
    lin = np.clip(lin * mod[..., None], 0.0, 1.0)
    return np.clip(np.round(np.power(lin, 1.0 / 2.2) * 255.0), 0, 255).astype(np.uint8)


def _tint_linear(patch_u8: np.ndarray, target_linear_rgb: Tuple[float, float, float], weight: float) -> np.ndarray:
    lin = np.power(patch_u8.astype(np.float32) / 255.0, 2.2)
    tint = np.array(target_linear_rgb, dtype=np.float32)[None, None, :]
    luma = np.mean(lin, axis=-1, keepdims=True)
    tinted = (1.0 - float(weight)) * lin + float(weight) * (0.4 + 0.9 * luma) * tint
    tinted = np.clip(tinted, 0.0, 1.0)
    return np.clip(np.round(np.power(tinted, 1.0 / 2.2) * 255.0), 0, 255).astype(np.uint8)


def _render_cleaved_patch(patch_u8: np.ndarray, angle_deg: float, half_gap_px: int = 7) -> np.ndarray:
    h, w = patch_u8.shape[:2]
    out = patch_u8.copy()
    phi = np.deg2rad(float(angle_deg))
    yy, xx = np.indices((h, w), dtype=np.float32)
    s_perp = -(xx - 0.5 * (w - 1)) * float(np.sin(phi)) + (yy - 0.5 * (h - 1)) * float(np.cos(phi))
    gap = np.abs(s_perp) <= float(half_gap_px)
    out[gap] = np.array([22, 18, 16], dtype=np.uint8)
    return out


class CausalExplorationGame:
    """
    Blind 5x5 multi-physics exploration game rendered with real 4K Yosemite textures.

    Supports predefined Level 1, Level 2 (rotated & dimmed), and custom blueprints
    synthesized by ELYSIA's CreativeSynthesisEngine for Level 3 self-verification.
    """

    GRID_ROWS = 5
    GRID_COLS = 5
    FOVEA_PX = 64
    TILE_PX = 40

    def __init__(
        self,
        level_id: int = 1,
        seed: int = 42,
        image_path: Path = YOSEMITE_4K_PATH,
        custom_blueprint: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.level_id = int(level_id)
        self._rng = np.random.default_rng(seed)

        if image_path.exists():
            yosemite = np.array(Image.open(image_path), dtype=np.uint8)
            floor_base = cv2.resize(yosemite[1620:1760, 980:1120], (self.FOVEA_PX, self.FOVEA_PX), interpolation=cv2.INTER_AREA)
            wall_base = cv2.resize(yosemite[1500:1640, 2800:2940], (self.FOVEA_PX, self.FOVEA_PX), interpolation=cv2.INTER_AREA)
            timber_base = cv2.resize(yosemite[1540:1680, 1520:1660], (self.FOVEA_PX, self.FOVEA_PX), interpolation=cv2.INTER_AREA)
            # Soften incidental rock strata on floor & pre-grain timber so matte floor stays isotropic
            floor_base = cv2.GaussianBlur(floor_base, (7, 7), 1.6)
            timber_base = cv2.GaussianBlur(timber_base, (7, 7), 1.6)
        else:
            yy, xx = np.indices((self.FOVEA_PX, self.FOVEA_PX), dtype=np.float32)
            base_tex = 110.0 + 14.0 * np.sin(0.25 * xx) * np.cos(0.22 * yy)
            floor_base = np.stack([base_tex, base_tex * 0.98, base_tex * 0.94], axis=-1).clip(0, 255).astype(np.uint8)
            wall_base = np.stack([base_tex * 0.78, base_tex * 0.80, base_tex * 0.82], axis=-1).clip(0, 255).astype(np.uint8)
            timber_base = np.stack([base_tex * 0.92, base_tex * 0.85, base_tex * 0.75], axis=-1).clip(0, 255).astype(np.uint8)
            floor_base = cv2.GaussianBlur(floor_base, (5, 5), 1.2)
            timber_base = cv2.GaussianBlur(timber_base, (5, 5), 1.2)

        self._floor_base = floor_base
        self._wall_base = wall_base
        self._timber_base = timber_base

        if custom_blueprint is not None:
            layout = custom_blueprint["layout"]
            self._start_pos = tuple(custom_blueprint["start"])
            self._goal_pos = tuple(custom_blueprint["goal"])
            self._grain_deg = float(custom_blueprint.get("grain_angle_deg", 0.0))
            self._illum_scale = float(custom_blueprint.get("illumination_scale", 1.0))
        elif self.level_id == 1:
            self._start_pos = (4, 0)
            self._goal_pos = (0, 4)
            self._grain_deg = 0.0  # Horizontal grain -> weak push North/South (90 deg)
            self._illum_scale = 1.0
            layout = [
                ["SAFE_FLOOR", "RIGID_WALL", "SUNLIT_SAFE_FLOOR", "SAFE_FLOOR", "GOAL_BEACON"],
                ["CRIMSON_HAZARD", "RIGID_WALL", "TIMBER_BARRIER", "RIGID_WALL", "CRIMSON_HAZARD"],
                ["SAFE_FLOOR", "DECEPTIVE_FOAM", "SAFE_FLOOR", "CRIMSON_HAZARD", "SAFE_FLOOR"],
                ["SAFE_FLOOR", "RIGID_WALL", "SUNLIT_SAFE_FLOOR", "RIGID_WALL", "SAFE_FLOOR"],
                ["SAFE_FLOOR", "SAFE_FLOOR", "DECEPTIVE_FOAM", "SAFE_FLOOR", "SAFE_FLOOR"],
            ]
        else:
            # Level 2: rotated grain (90 deg -> weak push East/West), 0.72x illumination dimming
            self._start_pos = (4, 4)
            self._goal_pos = (0, 0)
            self._grain_deg = 90.0
            self._illum_scale = 0.72
            layout = [
                ["GOAL_BEACON", "CRIMSON_HAZARD", "SAFE_FLOOR", "RIGID_WALL", "SAFE_FLOOR"],
                ["SUNLIT_SAFE_FLOOR", "RIGID_WALL", "DECEPTIVE_FOAM", "RIGID_WALL", "CRIMSON_HAZARD"],
                ["SAFE_FLOOR", "TIMBER_BARRIER", "SAFE_FLOOR", "SUNLIT_SAFE_FLOOR", "SAFE_FLOOR"],
                ["CRIMSON_HAZARD", "RIGID_WALL", "SAFE_FLOOR", "RIGID_WALL", "SAFE_FLOOR"],
                ["SAFE_FLOOR", "CRIMSON_HAZARD", "DECEPTIVE_FOAM", "SAFE_FLOOR", "SAFE_FLOOR"],
            ]

        self._grid: List[List[_GameTile]] = []
        for r in range(self.GRID_ROWS):
            row_tiles: List[_GameTile] = []
            for c in range(self.GRID_COLS):
                row_tiles.append(self._build_tile(r, c, layout[r][c]))
            self._grid.append(row_tiles)

        self.agent_pos: Tuple[int, int] = self._start_pos
        self.energy: float = 100.0
        self.level_cleared: bool = False
        self.step_count: int = 0
        self.hazard_hits: int = 0
        self.probes_count: int = 0
        self.cleaved_count: int = 0
        self.trajectory: List[Tuple[int, int]] = [self._start_pos]

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (self.GRID_ROWS, self.GRID_COLS)

    def _build_tile(self, r: int, c: int, kind: str) -> _GameTile:
        s = self._illum_scale
        if kind == "SAFE_FLOOR":
            # Muted natural rock floor (low chroma, isotropic matte stone well above dark walls)
            patch = _scale_linear_irradiance(_tint_linear(self._floor_base, (0.58, 0.58, 0.56), 0.45), 1.05 * s)
            return _GameTile(
                row=r, col=c, kind=kind, passable=True, is_hazard=False, is_goal=False,
                patch_rgb=patch, stiffness_n_m=9000.0, k_perp_n_m=9000.0,
                fy_par_n=85.0, fy_perp_n=85.0, damping_ratio=0.10, plastic_softening=0.0,
                peak_freq_hz=1800.0, decay_rate_s=22.0, quality_q=10.0,
            )
        if kind == "SUNLIT_SAFE_FLOOR":
            # High-luminance (+0.75 EV relative to SAFE_FLOOR) and strictly neutral low-saturation!
            patch = _scale_linear_irradiance(_tint_linear(self._floor_base, (0.92, 0.92, 0.92), 0.60), 1.65 * s)
            return _GameTile(
                row=r, col=c, kind=kind, passable=True, is_hazard=False, is_goal=False,
                patch_rgb=patch, stiffness_n_m=9200.0, k_perp_n_m=9200.0,
                fy_par_n=85.0, fy_perp_n=85.0, damping_ratio=0.10, plastic_softening=0.0,
                peak_freq_hz=1850.0, decay_rate_s=21.0, quality_q=10.0,
            )
        if kind == "CRIMSON_HAZARD":
            # Vivid crimson high-chroma hazard, moderate luminance
            patch = _scale_linear_irradiance(_tint_linear(self._floor_base, (0.95, 0.04, 0.03), 0.84), 0.85 * s)
            return _GameTile(
                row=r, col=c, kind=kind, passable=True, is_hazard=True, is_goal=False,
                patch_rgb=patch, stiffness_n_m=4200.0, k_perp_n_m=4200.0,
                fy_par_n=55.0, fy_perp_n=55.0, damping_ratio=0.25, plastic_softening=0.0,
                peak_freq_hz=620.0, decay_rate_s=50.0, quality_q=4.0,
            )
        if kind == "RIGID_WALL":
            # Dark neutral granite wall with crisp border inset
            patch = _scale_linear_irradiance(_tint_linear(self._wall_base, (0.32, 0.34, 0.36), 0.45), 0.42 * s)
            cv2.rectangle(patch, (3, 3), (self.FOVEA_PX - 4, self.FOVEA_PX - 4), (42, 44, 48), 2)
            return _GameTile(
                row=r, col=c, kind=kind, passable=False, is_hazard=False, is_goal=False,
                patch_rgb=patch, stiffness_n_m=12500.0, k_perp_n_m=12500.0,
                fy_par_n=90.0, fy_perp_n=90.0, damping_ratio=0.08, plastic_softening=0.0,
                peak_freq_hz=2450.0, decay_rate_s=18.0, quality_q=14.0,
            )
        if kind == "TIMBER_BARRIER":
            # Anisotropic striated timber barrier
            grained = _apply_oriented_grain(self._timber_base, angle_deg=self._grain_deg, wavelength_px=6.0, amp=0.55)
            patch = _scale_linear_irradiance(_tint_linear(grained, (0.58, 0.48, 0.34), 0.25), 0.95 * s)
            return _GameTile(
                row=r, col=c, kind=kind, passable=False, is_hazard=False, is_goal=False,
                patch_rgb=patch, stiffness_n_m=11000.0, k_perp_n_m=1700.0,
                fy_par_n=45.0, fy_perp_n=10.0, damping_ratio=0.22, plastic_softening=0.48,
                peak_freq_hz=1650.0, decay_rate_s=32.0, quality_q=8.5,
                phi_mech_deg=self._grain_deg, is_cleavable=True,
            )
        if kind == "DECEPTIVE_FOAM":
            # Visually looks like dark granite coarsely, reveals 2x2 micropores in 1:1 fovea, mechanically soft
            base_foam = _scale_linear_irradiance(_tint_linear(self._wall_base, (0.34, 0.36, 0.38), 0.40), 0.46 * s)
            patch = _apply_micropores(base_foam, amp=0.38)
            return _GameTile(
                row=r, col=c, kind=kind, passable=False, is_hazard=False, is_goal=False,
                patch_rgb=patch, stiffness_n_m=480.0, k_perp_n_m=480.0,
                fy_par_n=8.0, fy_perp_n=8.0, damping_ratio=0.72, plastic_softening=0.45,
                peak_freq_hz=300.0, decay_rate_s=105.0, quality_q=1.8,
                is_compressible=True,
            )
        if kind == "GOAL_BEACON":
            patch = _scale_linear_irradiance(_tint_linear(self._floor_base, (0.98, 0.85, 0.04), 0.88), 1.35 * s)
            cv2.circle(patch, (self.FOVEA_PX // 2, self.FOVEA_PX // 2), 14, (255, 240, 90), -1)
            return _GameTile(
                row=r, col=c, kind=kind, passable=True, is_hazard=False, is_goal=True,
                patch_rgb=patch, stiffness_n_m=7500.0, k_perp_n_m=7500.0,
                fy_par_n=75.0, fy_perp_n=75.0, damping_ratio=0.06, plastic_softening=0.0,
                peak_freq_hz=880.0, decay_rate_s=14.0, quality_q=12.0,
            )
        raise ValueError(f"Unknown tile kind: {kind}")

    def observe_frame(self) -> np.ndarray:
        """Render the full 200x200 RGB world frame with subtle agent marker."""
        canvas = np.zeros(
            (self.GRID_ROWS * self.TILE_PX, self.GRID_COLS * self.TILE_PX, 3), dtype=np.uint8
        )
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                tile = self._grid[r][c]
                resized = cv2.resize(tile.patch_rgb, (self.TILE_PX, self.TILE_PX), interpolation=cv2.INTER_AREA)
                y0, x0 = r * self.TILE_PX, c * self.TILE_PX
                canvas[y0 : y0 + self.TILE_PX, x0 : x0 + self.TILE_PX] = resized
                # Faint grid hairline separating tiles
                cv2.rectangle(canvas, (x0, y0), (x0 + self.TILE_PX - 1, y0 + self.TILE_PX - 1), (28, 28, 30), 1)

        ar, ac = self.agent_pos
        cy = ar * self.TILE_PX + self.TILE_PX // 2
        cx = ac * self.TILE_PX + self.TILE_PX // 2
        cv2.circle(canvas, (cx, cy), 9, (40, 170, 245), -1)
        cv2.circle(canvas, (cx, cy), 9, (255, 255, 255), 2)
        return canvas

    def foveate_tile(self, r: int, c: int) -> np.ndarray:
        """Return native 1:1 64x64 RGB crop of tile (r, c) without agent overlay."""
        if not (0 <= r < self.GRID_ROWS and 0 <= c < self.GRID_COLS):
            raise IndexError(f"Tile ({r}, {c}) out of bounds")
        return self._grid[r][c].patch_rgb.copy()

    def probe_push_tile(
        self, r: int, c: int, dir_yx: Tuple[float, float], force_n: float = 16.0
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Apply physical push to tile (r, c) along dir_yx=(dy, dx).
        Returns (force_t, disp_t, state_changed_passable).
        """
        if not (0 <= r < self.GRID_ROWS and 0 <= c < self.GRID_COLS):
            raise IndexError(f"Tile ({r}, {c}) out of bounds")
        self.probes_count += 1
        tile = self._grid[r][c]

        n_half = 40
        t_up = np.linspace(0.0, 1.0, n_half, dtype=np.float64)
        t_dn = np.linspace(1.0, 0.0, n_half, dtype=np.float64)
        force_t = np.concatenate([t_up, t_dn]) * float(force_n)

        if tile.phi_mech_deg is not None and not tile.cleaved:
            theta = float(np.arctan2(float(dir_yx[0]), float(dir_yx[1])) % np.pi)
            phi = float(np.deg2rad(tile.phi_mech_deg))
            delta = theta - phi
            cos2 = float(np.cos(delta) ** 2)
            sin2 = float(np.sin(delta) ** 2)
            k_theta = tile.stiffness_n_m * cos2 + tile.k_perp_n_m * sin2
            fy_theta = tile.fy_par_n * cos2 + tile.fy_perp_n * sin2
        else:
            k_theta = tile.stiffness_n_m
            fy_theta = tile.fy_par_n

        damp = tile.damping_ratio
        disp_list = []
        plastic_offset = 0.0
        max_f_seen = 0.0
        for idx, f_val in enumerate(force_t):
            is_loading = idx < n_half
            if f_val <= fy_theta:
                x_el = f_val / k_theta
            else:
                excess = f_val - fy_theta
                x_el = (fy_theta / k_theta) + (excess / (0.12 * k_theta))
                if is_loading and f_val > max_f_seen:
                    plastic_offset = max(plastic_offset, tile.plastic_softening * (excess / (0.12 * k_theta)))
            max_f_seen = max(max_f_seen, f_val)

            phase_frac = (idx / float(n_half)) if is_loading else ((2 * n_half - 1 - idx) / float(n_half))
            hyst = (
                (0.5 * damp * (force_n / k_theta) * phase_frac)
                if not is_loading
                else (-0.2 * damp * (force_n / k_theta) * phase_frac)
            )
            x_val = (f_val / k_theta) + plastic_offset + max(0.0, hyst) if not is_loading else max(0.0, x_el + hyst)
            disp_list.append(x_val)

        disp_t = np.asarray(disp_list, dtype=np.float64)
        disp_t += self._rng.normal(0.0, max(1e-6, 0.003 * np.max(disp_t)), size=disp_t.size)
        disp_t = np.maximum(0.0, disp_t)

        state_changed = False
        if not tile.passable and force_n > fy_theta:
            if tile.is_cleavable and tile.phi_mech_deg is not None:
                tile.cleaved = True
                tile.passable = True
                tile.kind = "CLEAVED_PASSAGE"
                tile.patch_rgb = _render_cleaved_patch(tile.patch_rgb, tile.phi_mech_deg)
                self.cleaved_count += 1
                state_changed = True
            elif tile.is_compressible:
                tile.passable = True
                tile.kind = "CLEARED_FOAM"
                tile.patch_rgb = _scale_linear_irradiance(_tint_linear(self._floor_base, (0.45, 0.45, 0.45), 0.35), 0.70 * self._illum_scale)
                state_changed = True

        return force_t.astype(np.float32), disp_t.astype(np.float32), state_changed

    def probe_tap_tile(self, r: int, c: int, impulse_n: float = 10.0) -> np.ndarray:
        """Return 4096-sample 16 kHz impact acoustic PCM waveform for tile (r, c)."""
        if not (0 <= r < self.GRID_ROWS and 0 <= c < self.GRID_COLS):
            raise IndexError(f"Tile ({r}, {c}) out of bounds")
        self.probes_count += 1
        tile = self._grid[r][c]
        sr = 16000
        n = 4096
        t = np.arange(n, dtype=np.float64) / float(sr)
        amp = 0.08 * float(impulse_n)
        env = np.exp(-tile.decay_rate_s * t)
        sig = amp * env * (
            np.sin(2.0 * np.pi * tile.peak_freq_hz * t)
            + 0.25 * np.sin(2.0 * np.pi * 1.58 * tile.peak_freq_hz * t)
        )
        noise_scale = amp * (0.35 / max(1.0, tile.quality_q))
        sig += np.exp(-150.0 * t) * self._rng.normal(0.0, noise_scale, size=n)
        return sig.astype(np.float32)

    def move(self, direction: str) -> GameStepOutcome:
        deltas = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
        if direction not in deltas:
            raise ValueError(f"Invalid direction '{direction}'")

        self.step_count += 1
        dr, dc = deltas[direction]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        if not (0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS):
            self.energy -= 1.0
            return GameStepOutcome(
                direction=direction,
                new_pos=self.agent_pos,
                moved=False,
                blocked=True,
                repelled=False,
                energy_delta=-1.0,
                energy_remaining=self.energy,
                level_cleared=self.level_cleared,
                frame_rgb=self.observe_frame(),
            )

        target = self._grid[nr][nc]
        if target.is_hazard:
            self.hazard_hits += 1
            self.energy -= 15.0
            return GameStepOutcome(
                direction=direction,
                new_pos=self.agent_pos,
                moved=False,
                blocked=False,
                repelled=True,
                energy_delta=-15.0,
                energy_remaining=self.energy,
                level_cleared=self.level_cleared,
                frame_rgb=self.observe_frame(),
            )

        if not target.passable:
            self.energy -= 1.0
            return GameStepOutcome(
                direction=direction,
                new_pos=self.agent_pos,
                moved=False,
                blocked=True,
                repelled=False,
                energy_delta=-1.0,
                energy_remaining=self.energy,
                level_cleared=self.level_cleared,
                frame_rgb=self.observe_frame(),
            )

        self.agent_pos = (nr, nc)
        self.trajectory.append(self.agent_pos)
        delta_e = 30.0 if target.is_goal else -0.5
        self.energy += delta_e
        if target.is_goal:
            self.level_cleared = True

        return GameStepOutcome(
            direction=direction,
            new_pos=self.agent_pos,
            moved=True,
            blocked=False,
            repelled=False,
            energy_delta=delta_e,
            energy_remaining=self.energy,
            level_cleared=self.level_cleared,
            frame_rgb=self.observe_frame(),
        )

    def evaluate_ground_truth_tile(self, r: int, c: int) -> Dict[str, Any]:
        """Read-only verification helper ONLY. Never accessed by explorer/learner agents."""
        t = self._grid[r][c]
        return {
            "row": t.row,
            "col": t.col,
            "kind": t.kind,
            "passable": t.passable,
            "is_hazard": t.is_hazard,
            "is_goal": t.is_goal,
            "phi_mech_deg": t.phi_mech_deg,
            "stiffness_n_m": t.stiffness_n_m,
            "k_perp_n_m": t.k_perp_n_m,
            "peak_freq_hz": t.peak_freq_hz,
            "cleaved": t.cleaved,
        }
