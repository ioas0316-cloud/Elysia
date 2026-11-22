"""
dynamic_spacetime_world.py

Continuous-Time Moving, Rotating, Shadow-Modulated 4K Multi-Physics World
========================================================================
Provides a strictly raw sensory interface (DynamicSpacetimeWorldProtocol) at any
continuous real timestamp t:
  - observe_frame_at(t) -> 135x240 periphery RGB array
  - foveate_at(t, y_norm, x_norm, fov_frac) -> native 1:1 high-res RGB crop
  - probe_push_at(t, y_norm, x_norm, force_n, dir_yx) -> (force_t, disp_t, post_periph_rgb)
  - probe_tap_at(t, y_norm, x_norm, impulse_n) -> 1D PCM waveform

Constructed from genuine 4K real-world camera captures:
  - genuine_optical_4k_yosemite.jpg (backdrop + striated granite-timber disks)
  - second_real_world_4k.jpg (isotropic rock disk)

Three non-colliding circular disks (radius = 85 px on 540x960 canvas) drift and
rotate smoothly over continuous time:
1. rotating_timber_A: phi_opt0 = 20 deg, omega = +30 deg/s, irrad(t) = 0.78 + 0.22*cos(pi*t),
   phi_mech(t) == phi_opt(t), k_par=11000, k_perp=1800, Fy_par=45, Fy_perp=10.
2. moving_granite_B: isotropic second_real_world_4k.jpg texture, omega = +20 deg/s,
   k_par=k_perp=12000, Fy=80.
3. rotating_veneer_C: phi_opt0 = 150 deg, omega = -25 deg/s, hidden internal bond lattice
   locked at orthogonal offset phi_mech(t) = (phi_opt(t) + 90) % 180.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np
from PIL import Image

from simulators.anisotropic_cleavage_world import (
    _apply_linear_irradiance_scale,
    _apply_oriented_grain,
)
from simulators.hidden_multiphysics_world import YOSEMITE_4K_PATH

SECOND_REAL_WORLD_4K_PATH = Path(r"c:\Elysia\demos\real_world_assets\second_real_world_4k.jpg")


class DynamicSpacetimeWorldProtocol(Protocol):
    """Strict blind sensory protocol at continuous timestamp t."""

    def observe_frame_at(self, t: float) -> np.ndarray:
        ...

    def foveate_at(
        self, t: float, y_norm: float, x_norm: float, fov_frac: float = 0.09
    ) -> np.ndarray:
        ...

    def probe_push_at(
        self,
        t: float,
        y_norm: float,
        x_norm: float,
        force_n: float = 18.0,
        dir_yx: Tuple[float, float] = (0.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...

    def probe_tap_at(
        self, t: float, y_norm: float, x_norm: float, impulse_n: float = 10.0
    ) -> np.ndarray:
        ...


@dataclass
class _MovingDiskBody:
    body_id: str
    radius_px: int
    cy0: float
    cx0: float
    vy_px_s: float
    vx_px_s: float
    phi_opt0_deg: Optional[float]
    phi_mech0_deg: float
    omega_deg_s: float
    patch_260_rgb: np.ndarray
    k_par_n_m: float
    k_perp_n_m: float
    fy_par_n: float
    fy_perp_n: float
    damping_ratio: float
    plastic_softening: float
    peak_freq_hz: float
    decay_rate_s: float
    quality_q: float
    irradiance_mod: bool = False
    is_isotropic: bool = False

    def center_at(self, t: float) -> Tuple[float, float]:
        return (self.cy0 + self.vy_px_s * float(t), self.cx0 + self.vx_px_s * float(t))

    def delta_angle_deg(self, t: float) -> float:
        return self.omega_deg_s * float(t)

    def phi_opt_at_deg(self, t: float) -> Optional[float]:
        if self.phi_opt0_deg is None:
            return None
        return float((self.phi_opt0_deg + self.omega_deg_s * float(t)) % 180.0)

    def phi_mech_at_deg(self, t: float) -> float:
        return float((self.phi_mech0_deg + self.omega_deg_s * float(t)) % 180.0)

    def irrad_at(self, t: float) -> float:
        if not self.irradiance_mod:
            return 1.0
        return float(0.78 + 0.22 * np.cos(np.pi * float(t)))


class DynamicSpacetimeWorld:
    """Continuous-time physical world over genuine 4K Yosemite + second real-world textures."""

    CANVAS_H = 540
    CANVAS_W = 960
    PERIPH_W = 240
    PERIPH_H = 135

    def __init__(
        self,
        yosemite_path: Path = YOSEMITE_4K_PATH,
        second_path: Path = SECOND_REAL_WORLD_4K_PATH,
        seed: int = 42,
    ) -> None:
        if not yosemite_path.exists():
            raise FileNotFoundError(f"4K photo not found: {yosemite_path}")
        self._rng = np.random.default_rng(seed)

        yosemite = np.array(Image.open(yosemite_path), dtype=np.uint8)
        self._backdrop_rgb = yosemite[0:540, 1600:2560].copy()

        base_260 = yosemite[1500:1760, 1500:1760].copy()
        timber_a_base = _apply_oriented_grain(base_260, angle_deg=20.0, wavelength_px=6.0, amp=0.35)
        veneer_c_base = _apply_oriented_grain(base_260, angle_deg=150.0, wavelength_px=6.0, amp=0.35)

        if second_path.exists():
            second_arr = np.array(Image.open(second_path), dtype=np.uint8)
            granite_b_base = second_arr[1350:1610, 2450:2710].copy()
        else:
            granite_b_base = yosemite[1500:1760, 2800:3060].copy()

        self._bodies: List[_MovingDiskBody] = [
            _MovingDiskBody(
                body_id="rotating_timber_A",
                radius_px=108,
                cy0=135.0,
                cx0=165.0,
                vy_px_s=14.0,
                vx_px_s=55.0,
                phi_opt0_deg=20.0,
                phi_mech0_deg=20.0,
                omega_deg_s=30.0,
                patch_260_rgb=timber_a_base,
                k_par_n_m=11000.0,
                k_perp_n_m=1800.0,
                fy_par_n=45.0,
                fy_perp_n=10.0,
                damping_ratio=0.22,
                plastic_softening=0.48,
                peak_freq_hz=1650.0,
                decay_rate_s=32.0,
                quality_q=8.5,
                irradiance_mod=True,
                is_isotropic=False,
            ),
            _MovingDiskBody(
                body_id="moving_granite_B",
                radius_px=108,
                cy0=405.0,
                cx0=200.0,
                vy_px_s=-8.0,
                vx_px_s=50.0,
                phi_opt0_deg=None,
                phi_mech0_deg=0.0,
                omega_deg_s=20.0,
                patch_260_rgb=granite_b_base,
                k_par_n_m=12000.0,
                k_perp_n_m=12000.0,
                fy_par_n=80.0,
                fy_perp_n=80.0,
                damping_ratio=0.08,
                plastic_softening=0.0,
                peak_freq_hz=2400.0,
                decay_rate_s=18.0,
                quality_q=14.0,
                irradiance_mod=False,
                is_isotropic=True,
            ),
            _MovingDiskBody(
                body_id="rotating_veneer_C",
                radius_px=108,
                cy0=275.0,
                cx0=770.0,
                vy_px_s=-14.0,
                vx_px_s=-50.0,
                phi_opt0_deg=150.0,
                phi_mech0_deg=60.0,  # (150 + 90) % 180 -> strictly orthogonal interior bonds
                omega_deg_s=-25.0,
                patch_260_rgb=veneer_c_base,
                k_par_n_m=11000.0,
                k_perp_n_m=1800.0,
                fy_par_n=45.0,
                fy_perp_n=10.0,
                damping_ratio=0.22,
                plastic_softening=0.48,
                peak_freq_hz=1600.0,
                decay_rate_s=32.0,
                quality_q=8.5,
                irradiance_mod=False,
                is_isotropic=False,
            ),
        ]

        # Precompute circular disk coordinates for fast compositing
        r = 108
        d = 2 * r + 1
        yy, xx = np.indices((d, d), dtype=np.float32)
        self._disk_r = r
        self._disk_mask = ((yy - r) ** 2 + (xx - r) ** 2) <= float(r * r)
        self._disk_dy = yy - r
        self._disk_dx = xx - r

    def _rotated_patch_crop(self, body: _MovingDiskBody, t: float) -> np.ndarray:
        """Rotate 260x260 patch clockwise by delta_angle_deg and crop centered (2r+1)x(2r+1)."""
        delta_deg = body.delta_angle_deg(t)
        # OpenCV positive angle is CCW; pass -delta_deg for clockwise rotation toward +y (dir_yx=(sin, cos))
        m_rot = cv2.getRotationMatrix2D((130.0, 130.0), -delta_deg, 1.0)
        rot_260 = cv2.warpAffine(
            body.patch_260_rgb,
            m_rot,
            (260, 260),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        r = body.radius_px
        crop = rot_260[130 - r : 130 + r + 1, 130 - r : 130 + r + 1]
        irrad = body.irrad_at(t)
        if abs(irrad - 1.0) > 1e-4:
            crop = _apply_linear_irradiance_scale(crop, irrad)
        return crop

    def _paste_disk(self, canvas: np.ndarray, body: _MovingDiskBody, t: float) -> None:
        cy_f, cx_f = body.center_at(t)
        cy = int(round(cy_f))
        cx = int(round(cx_f))
        r = body.radius_px
        patch = self._rotated_patch_crop(body, t)

        y0, y1 = cy - r, cy + r + 1
        x0, x1 = cx - r, cx + r + 1
        src_y0 = max(0, -y0)
        src_x0 = max(0, -x0)
        dst_y0 = max(0, y0)
        dst_x0 = max(0, x0)
        dst_y1 = min(self.CANVAS_H, y1)
        dst_x1 = min(self.CANVAS_W, x1)
        if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
            return
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        src_x1 = src_x0 + (dst_x1 - dst_x0)

        sub_mask = self._disk_mask[src_y0:src_y1, src_x0:src_x1]
        canvas[dst_y0:dst_y1, dst_x0:dst_x1][sub_mask] = patch[src_y0:src_y1, src_x0:src_x1][sub_mask]

    def _render_canvas_at(
        self,
        t: float,
        cleaved_body: Optional[_MovingDiskBody] = None,
        half_shift_px: int = 14,
    ) -> np.ndarray:
        canvas = self._backdrop_rgb.copy()
        for b in self._bodies:
            if b is not cleaved_body:
                self._paste_disk(canvas, b, t)

        if cleaved_body is not None:
            cy_f, cx_f = cleaved_body.center_at(t)
            cy = int(round(cy_f))
            cx = int(round(cx_f))
            r = cleaved_body.radius_px
            patch = self._rotated_patch_crop(cleaved_body, t)
            phi = float(np.deg2rad(cleaved_body.phi_mech_at_deg(t)))
            ny = float(np.cos(phi))
            nx = float(-np.sin(phi))

            s_perp = -self._disk_dx * float(np.sin(phi)) + self._disk_dy * float(np.cos(phi))
            gap_mask = self._disk_mask & (np.abs(s_perp) <= float(half_shift_px))
            gy_loc, gx_loc = np.where(gap_mask)
            g_dst_y = cy - r + gy_loc
            g_dst_x = cx - r + gx_loc
            g_val = (
                (g_dst_y >= 0)
                & (g_dst_y < self.CANVAS_H)
                & (g_dst_x >= 0)
                & (g_dst_x < self.CANVAS_W)
            )
            canvas[g_dst_y[g_val], g_dst_x[g_val]] = np.array([24, 20, 18], dtype=np.uint8)

            shift_y = int(round(half_shift_px * ny))
            shift_x = int(round(half_shift_px * nx))
            for sign, half_m in ((1, self._disk_mask & (s_perp >= 0.0)), (-1, self._disk_mask & (s_perp < 0.0))):
                py_l, px_l = np.where(half_m)
                dst_y = cy - r + py_l + sign * shift_y
                dst_x = cx - r + px_l + sign * shift_x
                ok = (dst_y >= 0) & (dst_y < self.CANVAS_H) & (dst_x >= 0) & (dst_x < self.CANVAS_W)
                canvas[dst_y[ok], dst_x[ok]] = patch[py_l[ok], px_l[ok]]

        return canvas

    def _find_body_at(self, t: float, y_norm: float, x_norm: float) -> Optional[_MovingDiskBody]:
        py = float(np.clip(y_norm, 0.0, 1.0) * (self.CANVAS_H - 1))
        px = float(np.clip(x_norm, 0.0, 1.0) * (self.CANVAS_W - 1))
        for b in self._bodies:
            cy, cx = b.center_at(t)
            if (py - cy) ** 2 + (px - cx) ** 2 <= float(b.radius_px ** 2):
                return b
        return None

    def observe_frame_at(self, t: float) -> np.ndarray:
        canvas = self._render_canvas_at(t)
        return cv2.resize(canvas, (self.PERIPH_W, self.PERIPH_H), interpolation=cv2.INTER_AREA)

    def render_full_canvas_at(
        self,
        t: float,
        cleave_body_id: Optional[str] = None,
    ) -> np.ndarray:
        """Read-only visualization helper rendering 540x960 frame (optional cleavage)."""
        target = next((b for b in self._bodies if b.body_id == cleave_body_id), None)
        return self._render_canvas_at(t, cleaved_body=target)

    def foveate_at(
        self, t: float, y_norm: float, x_norm: float, fov_frac: float = 0.09
    ) -> np.ndarray:
        canvas = self._render_canvas_at(t)
        crop_w = max(80, min(self.CANVAS_W, int(round(self.CANVAS_W * fov_frac))))
        crop_h = max(60, min(self.CANVAS_H, int(round(self.CANVAS_H * fov_frac))))
        cy = int(round(np.clip(y_norm, 0.0, 1.0) * self.CANVAS_H))
        cx = int(round(np.clip(x_norm, 0.0, 1.0) * self.CANVAS_W))
        y0 = int(np.clip(cy - crop_h // 2, 0, max(0, self.CANVAS_H - crop_h)))
        x0 = int(np.clip(cx - crop_w // 2, 0, max(0, self.CANVAS_W - crop_w)))
        return canvas[y0 : y0 + crop_h, x0 : x0 + crop_w].copy()

    def probe_push_at(
        self,
        t: float,
        y_norm: float,
        x_norm: float,
        force_n: float = 18.0,
        dir_yx: Tuple[float, float] = (0.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        body = self._find_body_at(t, y_norm, x_norm)

        n_half = 40
        t_up = np.linspace(0.0, 1.0, n_half, dtype=np.float64)
        t_dn = np.linspace(1.0, 0.0, n_half, dtype=np.float64)
        force_t = np.concatenate([t_up, t_dn]) * float(force_n)

        if body is None:
            disp_t = (force_t / 8.0) + self._rng.normal(0.0, 1e-4, size=force_t.size)
            disp_t = np.maximum(0.0, disp_t)
            return force_t.astype(np.float32), disp_t.astype(np.float32), self.observe_frame_at(t)

        theta = float(np.arctan2(float(dir_yx[0]), float(dir_yx[1])) % np.pi)
        phi = float(np.deg2rad(body.phi_mech_at_deg(t)))
        delta = theta - phi
        cos2 = float(np.cos(delta) ** 2)
        sin2 = float(np.sin(delta) ** 2)

        k_theta = body.k_par_n_m * cos2 + body.k_perp_n_m * sin2
        fy_theta = body.fy_par_n * cos2 + body.fy_perp_n * sin2
        damp = body.damping_ratio

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
                    plastic_offset = max(
                        plastic_offset, body.plastic_softening * (excess / (0.12 * k_theta))
                    )
            max_f_seen = max(max_f_seen, f_val)

            phase_frac = (idx / float(n_half)) if is_loading else ((2 * n_half - 1 - idx) / float(n_half))
            hyst = (
                (0.5 * damp * (force_n / k_theta) * phase_frac)
                if not is_loading
                else (-0.2 * damp * (force_n / k_theta) * phase_frac)
            )
            if not is_loading:
                x_val = (f_val / k_theta) + plastic_offset + max(0.0, hyst)
            else:
                x_val = max(0.0, x_el + hyst)
            disp_list.append(x_val)

        disp_t = np.asarray(disp_list, dtype=np.float64)
        disp_t += self._rng.normal(0.0, max(1e-6, 0.003 * np.max(disp_t)), size=disp_t.size)
        disp_t = np.maximum(0.0, disp_t)

        if force_n > fy_theta and body.plastic_softening > 0.0:
            cleaved = self._render_canvas_at(t, cleaved_body=body, half_shift_px=14)
            post_periph = cv2.resize(cleaved, (self.PERIPH_W, self.PERIPH_H), interpolation=cv2.INTER_AREA)
        else:
            post_periph = self.observe_frame_at(t)

        return force_t.astype(np.float32), disp_t.astype(np.float32), post_periph

    def probe_tap_at(
        self, t: float, y_norm: float, x_norm: float, impulse_n: float = 10.0
    ) -> np.ndarray:
        body = self._find_body_at(t, y_norm, x_norm)
        sr = 16000
        n = 4096
        tt = np.arange(n, dtype=np.float64) / float(sr)

        if body is None:
            return self._rng.normal(0.0, 1e-5, size=n).astype(np.float32)

        amp = 0.08 * float(impulse_n)
        env = np.exp(-body.decay_rate_s * tt)
        sig = amp * env * (
            np.sin(2.0 * np.pi * body.peak_freq_hz * tt)
            + 0.25 * np.sin(2.0 * np.pi * 1.58 * body.peak_freq_hz * tt)
        )
        noise_scale = amp * (0.35 / max(1.0, body.quality_q))
        sig += np.exp(-150.0 * tt) * self._rng.normal(0.0, noise_scale, size=n)
        return sig.astype(np.float32)

    def evaluate_ground_truth_at(
        self, t: float, y_norm: float, x_norm: float
    ) -> Optional[Dict[str, Any]]:
        """Read-only verification/test helper ONLY. Never called by cinema/explorer agents."""
        body = self._find_body_at(t, y_norm, x_norm)
        if body is None:
            return None
        cy, cx = body.center_at(t)
        return {
            "body_id": body.body_id,
            "t": float(t),
            "center_yx_norm": (float(cy / self.CANVAS_H), float(cx / self.CANVAS_W)),
            "phi_opt_deg": body.phi_opt_at_deg(t),
            "phi_mech_deg": body.phi_mech_at_deg(t),
            "omega_deg_s": body.omega_deg_s,
            "k_par_n_m": body.k_par_n_m,
            "k_perp_n_m": body.k_perp_n_m,
            "fy_par_n": body.fy_par_n,
            "fy_perp_n": body.fy_perp_n,
            "irradiance": body.irrad_at(t),
            "is_isotropic": body.is_isotropic,
        }
