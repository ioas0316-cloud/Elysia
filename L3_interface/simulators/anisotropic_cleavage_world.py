"""
anisotropic_cleavage_world.py

Anisotropic Bond-Lattice & Causal Cleavage Sensorimotor World
=============================================================
Builds a blind external environment conforming strictly to SensorimotorWorldProtocol
using genuine 4K Yosemite image textures (YOSEMITE_4K_PATH).

Four interior 180x180 physical bodies exhibit genuine visual-mechanical separation:
1. anisotropic_timber_A:
   - Visual striation grain & internal mechanical grain aligned at phi = 30 deg.
   - Strong along grain (k_par = 11000 N/m, Fy_par = 45 N), weak across grain
     (k_perp = 1800 N/m, Fy_perp = 10 N).
2. anisotropic_timber_B_rotated:
   - Same physical material rotated to phi = 120 deg and dimmed to 0.62x linear irradiance.
3. isotropic_granite_C:
   - True 4K natural texture without anisotropy; high isotropic stiffness (12000 N/m).
4. deceptive_veneer_D:
   - Surface optical veneer displays phi_opt = 0 deg, but hidden internal structural bonds
     run orthogonally at phi_mech = 90 deg (counter-example falsifying naive vision-only rules).

All hidden ground-truth states stay inside _AnisotropicBody and are NEVER exposed to
any explorer agent (evaluate_ground_truth_at is strictly for tests/verification scripts).
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
class _AnisotropicBody:
    body_id: str
    y0: int
    y1: int
    x0: int
    x1: int
    patch_rgb: np.ndarray
    phi_opt_deg: Optional[float]
    phi_mech_deg: float
    k_par_n_m: float
    k_perp_n_m: float
    fy_par_n: float
    fy_perp_n: float
    damping_ratio: float
    plastic_softening: float
    peak_freq_hz: float
    decay_rate_s: float
    quality_q: float
    irradiance_scale: float = 1.0
    is_isotropic: bool = False
    has_deceptive_veneer: bool = False

    def contains(self, py: int, px: int) -> bool:
        return self.y0 <= py < self.y1 and self.x0 <= px < self.x1

    @property
    def center_norm(self) -> Tuple[float, float]:
        return (0.5 * (self.y0 + self.y1) / 540.0, 0.5 * (self.x0 + self.x1) / 960.0)


def _apply_linear_irradiance_scale(patch_u8: np.ndarray, factor: float) -> np.ndarray:
    lin = np.power(patch_u8.astype(np.float32) / 255.0, 2.2)
    lin = np.clip(lin * float(factor), 0.0, 1.0)
    return np.clip(np.round(np.power(lin, 1.0 / 2.2) * 255.0), 0, 255).astype(np.uint8)


def _apply_oriented_grain(
    patch_u8: np.ndarray,
    angle_deg: float,
    wavelength_px: float = 6.0,
    amp: float = 0.30,
) -> np.ndarray:
    """
    Modulate linear-light irradiance with sinusoidal striations running ALONG angle_deg.

    Convention:
      theta = 0 deg  -> +x horizontal (dir_yx = (0, 1))
      theta = 90 deg -> +y downward   (dir_yx = (1, 0))
    Across-grain coordinate s_perp = -x * sin(phi) + y * cos(phi).
    """
    h, w = patch_u8.shape[:2]
    phi = np.deg2rad(float(angle_deg))
    yy, xx = np.indices((h, w), dtype=np.float32)
    s_perp = -xx * float(np.sin(phi)) + yy * float(np.cos(phi))
    mod = (1.0 + float(amp) * np.cos((2.0 * np.pi / float(wavelength_px)) * s_perp)).astype(np.float32)
    lin = np.power(patch_u8.astype(np.float32) / 255.0, 2.2)
    lin = np.clip(lin * mod[..., None], 0.0, 1.0)
    return np.clip(np.round(np.power(lin, 1.0 / 2.2) * 255.0), 0, 255).astype(np.uint8)


class AnisotropicCleavageWorld:
    """
    External physical world featuring directional grain stiffness, plastic yield,
    and physically causal bond-lattice cleavage across weak directions.
    """

    CANVAS_H = 540
    CANVAS_W = 960
    PERIPH_W = 240
    PERIPH_H = 135

    def __init__(self, image_path: Path = YOSEMITE_4K_PATH, seed: int = 42) -> None:
        if not image_path.exists():
            raise FileNotFoundError(f"4K photo not found: {image_path}")
        self._rng = np.random.default_rng(seed)
        yosemite = np.array(Image.open(image_path), dtype=np.uint8)

        # Pure smooth high-Sierra sky backdrop so all four interior bodies segment cleanly
        self._backdrop_rgb = yosemite[0:540, 1600:2560].copy()

        # Genuine 4K rock/forest surface patches (180x180) without luminance-cliff splits
        timber_base = yosemite[1540:1720, 1520:1700].copy()
        granite_base = yosemite[1500:1680, 2800:2980].copy()

        timber_a_patch = _apply_oriented_grain(timber_base, angle_deg=30.0, wavelength_px=6.0, amp=0.34)
        timber_b_patch = _apply_oriented_grain(
            _apply_linear_irradiance_scale(timber_base, 0.62),
            angle_deg=120.0,
            wavelength_px=6.0,
            amp=0.34,
        )
        granite_c_patch = granite_base.copy()
        veneer_d_patch = _apply_oriented_grain(timber_base, angle_deg=0.0, wavelength_px=6.0, amp=0.34)

        self._bodies: List[_AnisotropicBody] = [
            _AnisotropicBody(
                body_id="anisotropic_timber_A",
                y0=60,
                y1=240,
                x0=90,
                x1=270,
                patch_rgb=timber_a_patch,
                phi_opt_deg=30.0,
                phi_mech_deg=30.0,
                k_par_n_m=11000.0,
                k_perp_n_m=1800.0,
                fy_par_n=45.0,
                fy_perp_n=10.0,
                damping_ratio=0.22,
                plastic_softening=0.48,
                peak_freq_hz=1650.0,
                decay_rate_s=32.0,
                quality_q=8.5,
                irradiance_scale=1.0,
                is_isotropic=False,
                has_deceptive_veneer=False,
            ),
            _AnisotropicBody(
                body_id="anisotropic_timber_B_rotated",
                y0=60,
                y1=240,
                x0=650,
                x1=830,
                patch_rgb=timber_b_patch,
                phi_opt_deg=120.0,
                phi_mech_deg=120.0,
                k_par_n_m=11000.0,
                k_perp_n_m=1800.0,
                fy_par_n=45.0,
                fy_perp_n=10.0,
                damping_ratio=0.22,
                plastic_softening=0.48,
                peak_freq_hz=1620.0,
                decay_rate_s=32.0,
                quality_q=8.5,
                irradiance_scale=0.62,
                is_isotropic=False,
                has_deceptive_veneer=False,
            ),
            _AnisotropicBody(
                body_id="isotropic_granite_C",
                y0=300,
                y1=480,
                x0=90,
                x1=270,
                patch_rgb=granite_c_patch,
                phi_opt_deg=None,
                phi_mech_deg=0.0,
                k_par_n_m=12000.0,
                k_perp_n_m=12000.0,
                fy_par_n=80.0,
                fy_perp_n=80.0,
                damping_ratio=0.08,
                plastic_softening=0.0,
                peak_freq_hz=2400.0,
                decay_rate_s=18.0,
                quality_q=14.0,
                irradiance_scale=1.0,
                is_isotropic=True,
                has_deceptive_veneer=False,
            ),
            _AnisotropicBody(
                body_id="deceptive_veneer_D",
                y0=300,
                y1=480,
                x0=650,
                x1=830,
                patch_rgb=veneer_d_patch,
                phi_opt_deg=0.0,
                phi_mech_deg=90.0,
                k_par_n_m=11000.0,
                k_perp_n_m=1800.0,
                fy_par_n=45.0,
                fy_perp_n=10.0,
                damping_ratio=0.22,
                plastic_softening=0.48,
                peak_freq_hz=1600.0,
                decay_rate_s=32.0,
                quality_q=8.5,
                irradiance_scale=1.0,
                is_isotropic=False,
                has_deceptive_veneer=True,
            ),
        ]
        self._canvas_rgb = self._render_canvas()

    def _render_canvas(self) -> np.ndarray:
        canvas = self._backdrop_rgb.copy()
        for b in self._bodies:
            canvas[b.y0 : b.y1, b.x0 : b.x1] = b.patch_rgb
        return canvas

    def _render_cleaved_canvas(self, target_body: _AnisotropicBody, half_shift_px: int = 14) -> np.ndarray:
        """
        Render causal physical cleavage strictly along target_body's internal phi_mech plane.
        Two cohesive halves separate along normal (-sin(phi), cos(phi)) by +-half_shift_px,
        exposing an interior fracture crevice between the separated fragments.
        """
        canvas = self._backdrop_rgb.copy()
        for b in self._bodies:
            if b is not target_body:
                canvas[b.y0 : b.y1, b.x0 : b.x1] = b.patch_rgb

        bh = target_body.y1 - target_body.y0
        bw = target_body.x1 - target_body.x0
        phi = np.deg2rad(target_body.phi_mech_deg)
        # Normal to grain plane in (y, x) coordinates: (cos(phi), -sin(phi))
        ny = float(np.cos(phi))
        nx = float(-np.sin(phi))

        yy, xx = np.indices((bh, bw), dtype=np.float32)
        cy_loc = 0.5 * float(bh - 1)
        cx_loc = 0.5 * float(bw - 1)
        s_perp = -(xx - cx_loc) * float(np.sin(phi)) + (yy - cy_loc) * float(np.cos(phi))

        # Expose dark internal fracture fissure ONLY inside the cleaved gap band of original bbox
        gap_mask = np.abs(s_perp) <= float(half_shift_px)
        gy_idx, gx_idx = np.where(gap_mask)
        canvas[target_body.y0 + gy_idx, target_body.x0 + gx_idx] = np.array([24, 20, 18], dtype=np.uint8)

        shift_y_pos = int(round(half_shift_px * ny))
        shift_x_pos = int(round(half_shift_px * nx))

        for sign, mask in ((1, s_perp >= 0.0), (-1, s_perp < 0.0)):
            py_loc, px_loc = np.where(mask)
            dst_y = target_body.y0 + py_loc + sign * shift_y_pos
            dst_x = target_body.x0 + px_loc + sign * shift_x_pos
            valid = (dst_y >= 0) & (dst_y < self.CANVAS_H) & (dst_x >= 0) & (dst_x < self.CANVAS_W)
            canvas[dst_y[valid], dst_x[valid]] = target_body.patch_rgb[py_loc[valid], px_loc[valid]]

        return canvas

    def _find_body_at(self, y_norm: float, x_norm: float) -> Optional[_AnisotropicBody]:
        py = int(round(np.clip(y_norm, 0.0, 1.0) * (self.CANVAS_H - 1)))
        px = int(round(np.clip(x_norm, 0.0, 1.0) * (self.CANVAS_W - 1)))
        for b in self._bodies:
            if b.contains(py, px):
                return b
        return None

    def observe_periphery(self) -> np.ndarray:
        return cv2.resize(
            self._canvas_rgb, (self.PERIPH_W, self.PERIPH_H), interpolation=cv2.INTER_AREA
        )

    def foveate(self, y_norm: float, x_norm: float, fov_frac: float = 0.10) -> np.ndarray:
        crop_w = max(80, min(self.CANVAS_W, int(round(self.CANVAS_W * fov_frac))))
        crop_h = max(60, min(self.CANVAS_H, int(round(self.CANVAS_H * fov_frac))))
        cy = int(round(np.clip(y_norm, 0.0, 1.0) * self.CANVAS_H))
        cx = int(round(np.clip(x_norm, 0.0, 1.0) * self.CANVAS_W))
        y0 = int(np.clip(cy - crop_h // 2, 0, max(0, self.CANVAS_H - crop_h)))
        x0 = int(np.clip(cx - crop_w // 2, 0, max(0, self.CANVAS_W - crop_w)))
        return self._canvas_rgb[y0 : y0 + crop_h, x0 : x0 + crop_w].copy()

    def supports_physical_probing(self) -> bool:
        return True

    def mechanical_push(
        self,
        y_norm: float,
        x_norm: float,
        force_n: float = 18.0,
        dir_yx: Tuple[float, float] = (0.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        body = self._find_body_at(y_norm, x_norm)

        n_half = 40
        t_up = np.linspace(0.0, 1.0, n_half, dtype=np.float64)
        t_dn = np.linspace(1.0, 0.0, n_half, dtype=np.float64)
        force_t = np.concatenate([t_up, t_dn]) * float(force_n)

        if body is None:
            disp_t = (force_t / 8.0) + self._rng.normal(0.0, 1e-4, size=force_t.size)
            disp_t = np.maximum(0.0, disp_t)
            return force_t.astype(np.float32), disp_t.astype(np.float32), self.observe_periphery()

        # Probe angle theta in [0, pi) from dir_yx = (dy, dx)
        theta = float(np.arctan2(float(dir_yx[0]), float(dir_yx[1])) % np.pi)
        phi = float(np.deg2rad(body.phi_mech_deg))
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

        # Causal cleavage rupture if applied stress exceeds directional structural yield limit
        if force_n > fy_theta and body.plastic_softening > 0.0:
            cleaved_canvas = self._render_cleaved_canvas(body, half_shift_px=14)
            post_periph = cv2.resize(
                cleaved_canvas, (self.PERIPH_W, self.PERIPH_H), interpolation=cv2.INTER_AREA
            )
        else:
            post_periph = self.observe_periphery()

        return force_t.astype(np.float32), disp_t.astype(np.float32), post_periph

    def acoustic_tap(self, y_norm: float, x_norm: float, impulse_n: float = 10.0) -> np.ndarray:
        body = self._find_body_at(y_norm, x_norm)
        sr = 16000
        n = 4096
        t = np.arange(n, dtype=np.float64) / float(sr)

        if body is None:
            return self._rng.normal(0.0, 1e-5, size=n).astype(np.float32)

        amp = 0.08 * float(impulse_n)
        env = np.exp(-body.decay_rate_s * t)
        sig = amp * env * (
            np.sin(2.0 * np.pi * body.peak_freq_hz * t)
            + 0.25 * np.sin(2.0 * np.pi * 1.58 * body.peak_freq_hz * t)
        )
        noise_scale = amp * (0.35 / max(1.0, body.quality_q))
        sig += np.exp(-150.0 * t) * self._rng.normal(0.0, noise_scale, size=n)
        return sig.astype(np.float32)

    def evaluate_ground_truth_at(self, y_norm: float, x_norm: float) -> Optional[Dict[str, Any]]:
        """Read-only test/verification helper ONLY. Never accessed by explorer agents."""
        body = self._find_body_at(y_norm, x_norm)
        if body is None:
            return None
        return {
            "body_id": body.body_id,
            "center_norm": body.center_norm,
            "phi_opt_deg": body.phi_opt_deg,
            "phi_mech_deg": body.phi_mech_deg,
            "k_par_n_m": body.k_par_n_m,
            "k_perp_n_m": body.k_perp_n_m,
            "fy_par_n": body.fy_par_n,
            "fy_perp_n": body.fy_perp_n,
            "peak_freq_hz": body.peak_freq_hz,
            "irradiance_scale": body.irradiance_scale,
            "is_isotropic": body.is_isotropic,
            "has_deceptive_veneer": body.has_deceptive_veneer,
        }
