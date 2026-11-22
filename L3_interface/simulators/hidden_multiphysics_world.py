"""
hidden_multiphysics_world.py

External Real-Photo & Hidden Multi-Physics Sensorimotor Environments
====================================================================
Provides strict Agent-World boundary environments built over the genuine 4K
Yosemite photograph (genuine_optical_4k_yosemite.jpg).

The explorer interacts strictly through SensorimotorWorldProtocol:
  - observe_periphery() -> wide low-res RGB frame
  - foveate(y_norm, x_norm, fov_frac) -> native 1:1 high-res RGB crop
  - supports_physical_probing() -> bool
  - mechanical_push(...) -> (force_t, disp_t, post_push_periphery_rgb)
  - acoustic_tap(...) -> 1D PCM float32 waveform

Hidden physics states and labels are encapsulated inside the environment and
are NEVER exposed to the explorer agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

YOSEMITE_4K_PATH = Path(r"c:\Elysia\demos\real_world_assets\genuine_optical_4k_yosemite.jpg")


class SensorimotorWorldProtocol(Protocol):
    """Strict raw-sensory-array interface between agent and external world."""

    def observe_periphery(self) -> np.ndarray:
        ...

    def foveate(self, y_norm: float, x_norm: float, fov_frac: float = 0.12) -> np.ndarray:
        ...

    def supports_physical_probing(self) -> bool:
        ...

    def mechanical_push(
        self,
        y_norm: float,
        x_norm: float,
        force_n: float = 15.0,
        dir_yx: Tuple[float, float] = (0.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...

    def acoustic_tap(
        self, y_norm: float, x_norm: float, impulse_n: float = 10.0
    ) -> np.ndarray:
        ...


class Real4KYosemiteWorld:
    """Purely optical environment backed directly by the genuine 3840x2160 4K camera image."""

    def __init__(self, image_path: Path = YOSEMITE_4K_PATH, periphery_width: int = 240):
        if not image_path.exists():
            raise FileNotFoundError(f"4K photo not found: {image_path}")
        self._full_rgb = np.array(Image.open(image_path), dtype=np.uint8)
        self._h, self._w = self._full_rgb.shape[:2]
        new_h = max(32, int(round(self._h * (periphery_width / float(self._w)))))
        self._periphery_rgb = cv2.resize(
            self._full_rgb, (periphery_width, new_h), interpolation=cv2.INTER_AREA
        )

    @property
    def full_shape_hw(self) -> Tuple[int, int]:
        return (self._h, self._w)

    def observe_periphery(self) -> np.ndarray:
        return self._periphery_rgb.copy()

    def foveate(
        self, y_norm: float, x_norm: float, fov_frac: float = 0.10
    ) -> np.ndarray:
        crop_w = max(96, min(self._w, int(round(self._w * fov_frac))))
        crop_h = max(64, min(self._h, int(round(self._h * fov_frac))))
        cy = int(round(np.clip(y_norm, 0.0, 1.0) * self._h))
        cx = int(round(np.clip(x_norm, 0.0, 1.0) * self._w))
        y0 = int(np.clip(cy - crop_h // 2, 0, max(0, self._h - crop_h)))
        x0 = int(np.clip(cx - crop_w // 2, 0, max(0, self._w - crop_w)))
        return self._full_rgb[y0 : y0 + crop_h, x0 : x0 + crop_w].copy()

    def supports_physical_probing(self) -> bool:
        return False

    def mechanical_push(
        self,
        y_norm: float,
        x_norm: float,
        force_n: float = 15.0,
        dir_yx: Tuple[float, float] = (0.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("Optical photograph does not support mechanical contact probing.")

    def acoustic_tap(
        self, y_norm: float, x_norm: float, impulse_n: float = 10.0
    ) -> np.ndarray:
        raise NotImplementedError("Optical photograph does not support acoustic tapping.")


@dataclass
class _HiddenBody:
    body_id: str
    y0: int
    y1: int
    x0: int
    x1: int
    patch_rgb: np.ndarray
    stiffness_n_m: float
    damping_ratio: float
    yield_force_n: float
    plastic_softening: float
    movable: bool
    peak_freq_hz: float
    decay_rate_s: float
    quality_q: float
    has_cast_shadow: bool = False
    is_deceptive_micropore: bool = False

    def contains(self, py: int, px: int) -> bool:
        return self.y0 <= py < self.y1 and self.x0 <= px < self.x1

    @property
    def center_norm(self) -> Tuple[float, float]:
        return (0.5 * (self.y0 + self.y1) / 540.0, 0.5 * (self.x0 + self.x1) / 960.0)


def _apply_linear_shadow_right_half(patch_u8: np.ndarray, dark_factor: float = 0.45) -> np.ndarray:
    h, w = patch_u8.shape[:2]
    mask = np.ones(w, dtype=np.float32)
    mask[w // 2 :] = dark_factor
    soft = gaussian_filter1d(mask, sigma=1.5)
    lin = np.power(patch_u8.astype(np.float32) / 255.0, 2.2)
    lin *= soft[None, :, None]
    return np.clip(np.round(np.power(np.clip(lin, 0.0, 1.0), 1.0 / 2.2) * 255.0), 0, 255).astype(np.uint8)


def _apply_micropore_modulation(patch_u8: np.ndarray, amp: float = 0.30) -> np.ndarray:
    """Apply 4px-period (2x2 block) checkerboard in linear light (zero average in 4x4 periphery)."""
    h, w = patch_u8.shape[:2]
    yy, xx = np.indices((h, w))
    sign = 1.0 - 2.0 * (((yy // 2) + (xx // 2)) % 2)
    mod = (1.0 + amp * sign).astype(np.float32)
    lin = np.power(patch_u8.astype(np.float32) / 255.0, 2.2)
    lin = np.clip(lin * mod[..., None], 0.0, 1.0)
    return np.clip(np.round(np.power(lin, 1.0 / 2.2) * 255.0), 0, 255).astype(np.uint8)


class HiddenMultiPhysicsWorld:
    """
    Simulated multi-modal physical world rendered strictly with genuine 4K
    Yosemite surface textures on a 540x960 high-res canvas (135x240 periphery).
    """

    CANVAS_H = 540
    CANVAS_W = 960
    PERIPH_W = 240
    PERIPH_H = 135

    def __init__(self, image_path: Path = YOSEMITE_4K_PATH, seed: int = 42):
        if not image_path.exists():
            raise FileNotFoundError(f"4K photo not found: {image_path}")
        self._rng = np.random.default_rng(seed)
        yosemite = np.array(Image.open(image_path), dtype=np.uint8)

        # Smooth bright sky-mist tile backdrop (low internal texture)
        self._backdrop_rgb = yosemite[0:540, 960:1920].copy()

        # Construct 5 physical bodies strictly inside interior bounds
        slab_a_rgb = yosemite[1620:1780, 960:1140].copy()  # 160x180 sunlit granite
        slab_b_raw = yosemite[1620:1790, 1800:2040].copy()  # 170x240 bright uniform sunlit granite
        slab_b_rgb = _apply_linear_shadow_right_half(slab_b_raw, dark_factor=0.45)
        canopy_c_rgb = yosemite[1150:1310, 300:480].copy()  # 160x180 dark forest
        canopy_d_rgb = yosemite[1150:1310, 3000:3180].copy()  # 160x180 distant forest
        foam_e_raw = yosemite[1620:1780, 960:1160].copy()  # 160x200 same granite look
        foam_e_rgb = _apply_micropore_modulation(foam_e_raw, amp=0.35)

        self._bodies: List[_HiddenBody] = [
            _HiddenBody(
                body_id="rigid_slab_A",
                y0=70,
                y1=230,
                x0=90,
                x1=270,
                patch_rgb=slab_a_rgb,
                stiffness_n_m=12000.0,
                damping_ratio=0.08,
                yield_force_n=80.0,
                plastic_softening=0.0,
                movable=True,
                peak_freq_hz=2400.0,
                decay_rate_s=18.0,
                quality_q=14.0,
            ),
            _HiddenBody(
                body_id="rigid_slab_B_shadowed",
                y0=70,
                y1=240,
                x0=350,
                x1=590,
                patch_rgb=slab_b_rgb,
                stiffness_n_m=11500.0,
                damping_ratio=0.09,
                yield_force_n=80.0,
                plastic_softening=0.0,
                movable=True,
                peak_freq_hz=2350.0,
                decay_rate_s=20.0,
                quality_q=13.5,
                has_cast_shadow=True,
            ),
            _HiddenBody(
                body_id="compliant_canopy_C",
                y0=300,
                y1=460,
                x0=90,
                x1=270,
                patch_rgb=canopy_c_rgb,
                stiffness_n_m=650.0,
                damping_ratio=0.65,
                yield_force_n=40.0,
                plastic_softening=0.0,
                movable=False,
                peak_freq_hz=320.0,
                decay_rate_s=95.0,
                quality_q=2.0,
            ),
            _HiddenBody(
                body_id="compliant_canopy_D",
                y0=70,
                y1=230,
                x0=690,
                x1=870,
                patch_rgb=canopy_d_rgb,
                stiffness_n_m=700.0,
                damping_ratio=0.62,
                yield_force_n=40.0,
                plastic_softening=0.0,
                movable=False,
                peak_freq_hz=340.0,
                decay_rate_s=90.0,
                quality_q=2.1,
            ),
            _HiddenBody(
                body_id="deceptive_foam_E",
                y0=300,
                y1=460,
                x0=650,
                x1=850,
                patch_rgb=foam_e_rgb,
                stiffness_n_m=480.0,
                damping_ratio=0.72,
                yield_force_n=8.0,
                plastic_softening=0.45,
                movable=False,
                peak_freq_hz=290.0,
                decay_rate_s=110.0,
                quality_q=1.8,
                is_deceptive_micropore=True,
            ),
        ]
        self._canvas_rgb = self._render_canvas()

    def _render_canvas(self, shift_overrides: Optional[Dict[str, Tuple[int, int]]] = None) -> np.ndarray:
        canvas = self._backdrop_rgb.copy()
        for b in self._bodies:
            dy, dx = shift_overrides.get(b.body_id, (0, 0)) if shift_overrides else (0, 0)
            y0 = int(np.clip(b.y0 + dy, 0, self.CANVAS_H - (b.y1 - b.y0)))
            x0 = int(np.clip(b.x0 + dx, 0, self.CANVAS_W - (b.x1 - b.x0)))
            canvas[y0 : y0 + (b.y1 - b.y0), x0 : x0 + (b.x1 - b.x0)] = b.patch_rgb
        return canvas

    def _find_body_at(self, y_norm: float, x_norm: float) -> Optional[_HiddenBody]:
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

    def foveate(
        self, y_norm: float, x_norm: float, fov_frac: float = 0.14
    ) -> np.ndarray:
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
        force_n: float = 15.0,
        dir_yx: Tuple[float, float] = (0.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        body = self._find_body_at(y_norm, x_norm)

        # Generate loading-unloading force ramp (80 samples at dt=0.002s)
        n_half = 40
        t_up = np.linspace(0.0, 1.0, n_half, dtype=np.float64)
        t_dn = np.linspace(1.0, 0.0, n_half, dtype=np.float64)
        force_t = np.concatenate([t_up, t_dn]) * float(force_n)

        if body is None:
            # Probed open background air -> near-zero resistance
            disp_t = (force_t / 8.0) + self._rng.normal(0.0, 1e-4, size=force_t.size)
            disp_t = np.maximum(0.0, disp_t)
            return force_t.astype(np.float32), disp_t.astype(np.float32), self.observe_periphery()

        k = body.stiffness_n_m
        y_force = body.yield_force_n
        damp = body.damping_ratio

        disp_list = []
        plastic_offset = 0.0
        max_f_seen = 0.0
        for idx, f_val in enumerate(force_t):
            is_loading = idx < n_half
            if f_val <= y_force:
                x_el = f_val / k
            else:
                excess = f_val - y_force
                x_el = (y_force / k) + (excess / (0.14 * k))
                if is_loading and f_val > max_f_seen:
                    plastic_offset = max(plastic_offset, body.plastic_softening * (excess / (0.14 * k)))
            max_f_seen = max(max_f_seen, f_val)

            # Viscoelastic hysteresis widening between loading vs unloading halves
            phase_frac = (idx / float(n_half)) if is_loading else ((2 * n_half - 1 - idx) / float(n_half))
            hyst = (0.5 * damp * (force_n / k) * phase_frac) if not is_loading else (-0.2 * damp * (force_n / k) * phase_frac)
            if not is_loading:
                x_val = (f_val / k) + plastic_offset + max(0.0, hyst)
            else:
                x_val = max(0.0, x_el + hyst)
            disp_list.append(x_val)

        disp_t = np.asarray(disp_list, dtype=np.float64)
        disp_t += self._rng.normal(0.0, max(1e-6, 0.004 * np.max(disp_t)), size=disp_t.size)
        disp_t = np.maximum(0.0, disp_t)

        # Rigid body translation snapshot if body is movable and force >= 10N
        if body.movable and force_n >= 10.0:
            norm_d = float(np.hypot(dir_yx[0], dir_yx[1])) + 1e-8
            shift_y = int(round((dir_yx[0] / norm_d) * 40.0))
            shift_x = int(round((dir_yx[1] / norm_d) * 40.0))
            shifted_canvas = self._render_canvas(shift_overrides={body.body_id: (shift_y, shift_x)})
            post_periph = cv2.resize(
                shifted_canvas, (self.PERIPH_W, self.PERIPH_H), interpolation=cv2.INTER_AREA
            )
        else:
            post_periph = self.observe_periphery()

        return force_t.astype(np.float32), disp_t.astype(np.float32), post_periph

    def acoustic_tap(
        self, y_norm: float, x_norm: float, impulse_n: float = 10.0
    ) -> np.ndarray:
        body = self._find_body_at(y_norm, x_norm)
        sr = 16000
        n = 4096
        t = np.arange(n, dtype=np.float64) / float(sr)

        if body is None:
            return self._rng.normal(0.0, 1e-5, size=n).astype(np.float32)

        amp = 0.08 * float(impulse_n)
        env = np.exp(-body.decay_rate_s * t)
        # Sharp modal ring (high Q) vs broadband damped thud (low Q)
        sig = amp * env * (
            np.sin(2.0 * np.pi * body.peak_freq_hz * t)
            + 0.25 * np.sin(2.0 * np.pi * 1.58 * body.peak_freq_hz * t)
        )
        # Add realistic impact transient scaled inversely with Q
        noise_scale = amp * (0.35 / max(1.0, body.quality_q))
        sig += np.exp(-150.0 * t) * self._rng.normal(0.0, noise_scale, size=n)
        return sig.astype(np.float32)

    def evaluate_ground_truth_at(self, y_norm: float, x_norm: float) -> Optional[Dict[str, Any]]:
        """Read-only verification/test helper ONLY. Never called by explorer agent."""
        body = self._find_body_at(y_norm, x_norm)
        if body is None:
            return None
        return {
            "body_id": body.body_id,
            "center_norm": body.center_norm,
            "stiffness_n_m": body.stiffness_n_m,
            "damping_ratio": body.damping_ratio,
            "yield_force_n": body.yield_force_n,
            "movable": body.movable,
            "peak_freq_hz": body.peak_freq_hz,
            "has_cast_shadow": body.has_cast_shadow,
            "is_deceptive_micropore": body.is_deceptive_micropore,
        }
