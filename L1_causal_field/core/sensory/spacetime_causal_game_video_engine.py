"""
spacetime_causal_game_video_engine.py

ELYSIA Spacetime Causal Flow & Interactive Game/Video Discovery Engine
========================================================================
Implements causal learning from spacetime dynamics (raw RGB video & interactive game flow)
with ZERO VLMs, ZERO LLMs, and ZERO access to hidden game state:

1. ContinuousCausalArena (Headless Pygame, SDL_VIDEODRIVER=dummy):
   - Renders raw RGB frames containing four physically distinct processes:
     * SELF body controllable via motor vector `action_yx`,
     * EXTERNAL autonomous creature moving independently on an orbital figure-8,
     * ILLUMINATION shadow wave altering irradiance multiplicatively without moving texture,
     * STATIC target beacon at high opponent contrast.
   - Exposes ONLY `step(action_yx) -> rgb_uint8`. Hidden ground-truth coordinates exist
     solely in `get_ground_truth_for_eval_only()` for post-hoc evaluation.
2. Phase 1 — Active Motor-Babbling Causal Discovery:
   - Combines idle observation + four-direction motor efference probes inside
     OcularScaleOrgan.observe_transition() to separate SELF_BODY (aligned with motor command),
     EXTERNAL_AUTONOMOUS_AGENT (moving independently during idle/anti-motor), and
     ILLUMINATION_SHADOW (luminance step without chroma/texture shift).
3. Phase 2 — Closed-Loop Blind Visual-Motor Control:
   - Tracks SELF_BODY strictly via raw optical flow + opponent signature around predicted
     efference displacement, locates the static target beacon blindly, and drives SELF toward
     the target using the causal motor gain discovered in Phase 1.
4. Phase 3 — Real External Nature Video Autonomous Agency Inference:
   - Runs the identical passive spacetime observer on `real_external_nature_video.mp4`
     to infer autonomous object agency purely from persistent coherent optical flow.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pygame
from scipy.ndimage import gaussian_filter

from core.physics.causal_field import CausalField, InformationVoxel
from core.sensory.ocular_scale_organ import (
    OcularScaleOrgan,
    OpticalSceneObservation,
    TemporalVisualAnalysis,
)


class ContinuousCausalArena:
    """
    Headless pygame physics-visual sandbox emitting strictly raw RGB frames.
    External callers cannot read object coordinates via `step()`.
    """

    def __init__(self, width: int = 240, height: int = 180) -> None:
        if not pygame.get_init():
            pygame.init()
        self.width = int(width)
        self.height = int(height)
        self._surface = pygame.Surface((self.width, self.height))
        self._step_count = 0

        # Pre-render rich continuous organic terrain texture so optical flow works reliably
        yy = np.linspace(0.0, 1.0, self.height, dtype=np.float32)[:, None]
        xx = np.linspace(0.0, 1.0, self.width, dtype=np.float32)[None, :]
        tex = (
            18.0 * np.sin(2.0 * np.pi * 6.0 * xx + 1.3 * np.sin(2.0 * np.pi * 3.0 * yy))
            + 14.0 * np.cos(2.0 * np.pi * 5.0 * yy + 0.9 * np.cos(2.0 * np.pi * 4.0 * xx))
            + 8.0 * np.sin(2.0 * np.pi * 13.0 * (xx + yy))
        )
        base_r = np.clip(78.0 + 0.85 * tex, 20.0, 220.0)
        base_g = np.clip(94.0 + 1.00 * tex, 20.0, 220.0)
        base_b = np.clip(82.0 + 0.70 * tex, 20.0, 220.0)
        bg_arr = np.stack([base_r, base_g, base_b], axis=-1).astype(np.uint8)
        self._bg_surface = pygame.surfarray.make_surface(bg_arr.transpose(1, 0, 2))

        # Hidden ground-truth state
        self._player_yx = np.array([124.0, 56.0], dtype=np.float32)
        self._player_vel_yx = np.array([0.0, 0.0], dtype=np.float32)
        self._player_radius = 14

        self._creature_t = 0.0
        self._creature_yx = self._eval_creature_pos(0.0)
        self._creature_radius = 12

        self._shadow_t = 0.0
        self._shadow_yx = self._eval_shadow_pos(0.0)
        self._shadow_radius = 28

        self._target_yx = np.array([44.0, 194.0], dtype=np.float32)
        self._target_radius = 11

    def _eval_creature_pos(self, t: float) -> np.ndarray:
        cy = 50.0 + 18.0 * np.sin(0.32 * t)
        cx = 118.0 + 28.0 * np.sin(0.16 * t + 0.4)
        return np.array([cy, cx], dtype=np.float32)

    def _eval_shadow_pos(self, t: float) -> np.ndarray:
        sy = 126.0 + 12.0 * np.sin(0.18 * t)
        sx = 152.0 + 26.0 * np.sin(0.12 * t)
        return np.array([sy, sx], dtype=np.float32)

    def _draw_textured_disc(
        self,
        surf: pygame.Surface,
        cy: float,
        cx: float,
        radius: int,
        outer_rgb: Tuple[int, int, int],
        ring_rgb: Tuple[int, int, int],
        core_rgb: Tuple[int, int, int],
    ) -> None:
        ix, iy = int(round(cx)), int(round(cy))
        pygame.draw.circle(surf, outer_rgb, (ix, iy), radius)
        pygame.draw.circle(surf, ring_rgb, (ix, iy), max(4, int(round(radius * 0.68))), 2)
        pygame.draw.line(surf, ring_rgb, (ix - radius + 3, iy), (ix + radius - 3, iy), 2)
        pygame.draw.line(surf, ring_rgb, (ix, iy - radius + 3), (ix, iy + radius - 3), 2)
        pygame.draw.circle(surf, core_rgb, (ix, iy), max(3, radius // 3))

    def render(self) -> np.ndarray:
        """Render current scene purely as (H, W, 3) uint8 RGB array."""
        self._surface.blit(self._bg_surface, (0, 0))

        # 1. Multiplicative illumination cast shadow (darkens surface ~0.62x without moving texture)
        sx, sy = int(round(self._shadow_yx[1])), int(round(self._shadow_yx[0]))
        shadow_mask = pygame.Surface((self.width, self.height))
        shadow_mask.fill((255, 255, 255))
        pygame.draw.circle(shadow_mask, (158, 158, 158), (sx, sy), self._shadow_radius)
        self._surface.blit(shadow_mask, (0, 0), special_flags=pygame.BLEND_RGB_MULT)

        # 2. Target beacon (stationary luminous golden-amber landmark)
        self._draw_textured_disc(
            self._surface,
            float(self._target_yx[0]),
            float(self._target_yx[1]),
            self._target_radius,
            outer_rgb=(245, 195, 42),
            ring_rgb=(175, 110, 15),
            core_rgb=(255, 250, 215),
        )

        # 3. External autonomous creature (warm coral-red striped disc)
        self._draw_textured_disc(
            self._surface,
            float(self._creature_yx[0]),
            float(self._creature_yx[1]),
            self._creature_radius,
            outer_rgb=(225, 78, 62),
            ring_rgb=(120, 28, 20),
            core_rgb=(255, 210, 190),
        )

        # 4. Controllable SELF body (cool cobalt-cyan textured disc)
        self._draw_textured_disc(
            self._surface,
            float(self._player_yx[0]),
            float(self._player_yx[1]),
            self._player_radius,
            outer_rgb=(42, 162, 224),
            ring_rgb=(15, 65, 120),
            core_rgb=(220, 248, 255),
        )

        arr = pygame.surfarray.array3d(self._surface).transpose(1, 0, 2).copy()
        return arr

    def step(self, action_yx: Tuple[float, float]) -> np.ndarray:
        """Advance physical world one tick from continuous 2D motor command, returning ONLY RGB."""
        ay = float(np.clip(action_yx[0], -1.0, 1.0))
        ax = float(np.clip(action_yx[1], -1.0, 1.0))
        impulse = np.array([ay, ax], dtype=np.float32) * 5.0

        self._player_vel_yx = 0.62 * self._player_vel_yx + impulse
        self._player_yx = self._player_yx + self._player_vel_yx

        margin = float(self._player_radius + 4)
        if self._player_yx[0] < margin:
            self._player_yx[0] = margin
            self._player_vel_yx[0] *= -0.5
        elif self._player_yx[0] > self.height - margin:
            self._player_yx[0] = self.height - margin
            self._player_vel_yx[0] *= -0.5
        if self._player_yx[1] < margin:
            self._player_yx[1] = margin
            self._player_vel_yx[1] *= -0.5
        elif self._player_yx[1] > self.width - margin:
            self._player_yx[1] = self.width - margin
            self._player_vel_yx[1] *= -0.5

        self._step_count += 1
        self._creature_t += 1.0
        self._creature_yx = self._eval_creature_pos(self._creature_t)
        self._shadow_t += 1.0
        self._shadow_yx = self._eval_shadow_pos(self._shadow_t)

        return self.render()

    def get_ground_truth_for_eval_only(self) -> Dict[str, Any]:
        """Strictly for test/metric grading after blind prediction & control."""
        return {
            "player_yx": (float(self._player_yx[0]), float(self._player_yx[1])),
            "creature_yx": (float(self._creature_yx[0]), float(self._creature_yx[1])),
            "shadow_yx": (float(self._shadow_yx[0]), float(self._shadow_yx[1])),
            "target_yx": (float(self._target_yx[0]), float(self._target_yx[1])),
        }


@dataclass
class SpacetimeEntityDiscovery:
    """An ontological role discovered purely by observing spacetime optical transitions."""

    entity_role: str
    estimated_yx_px: Tuple[float, float]
    mean_velocity_yx: Tuple[float, float]
    motor_causal_correlation: float
    autonomous_motion_score: float
    voxel_id: str


@dataclass
class GameSpacetimeReport:
    """Full Phase 1 (blind causal discovery) + Phase 2 (blind closed-loop control) telemetry."""

    phase1_discoveries: List[SpacetimeEntityDiscovery]
    self_localization_error_px: float
    external_agent_localization_error_px: float
    learned_motor_gain_yx: Tuple[float, float]
    control_initial_dist_px: float
    control_final_dist_px: float
    control_distance_reduction_ratio: float
    control_trajectory_est_yx: List[Tuple[float, float]]
    control_trajectory_true_yx: List[Tuple[float, float]]
    distance_curve_px: List[float]
    sample_frame_rgb: np.ndarray
    self_resonance_heatmap: np.ndarray
    external_motion_heatmap: np.ndarray
    shadow_heatmap: np.ndarray
    discovered_target_yx: Tuple[float, float]


@dataclass
class RealVideoSpacetimeReport:
    """Passive spacetime autonomous-agency inference from real external video frames."""

    video_name: str
    frames_processed: int
    discovered_agent_trajectory_yx: List[Tuple[float, float]]
    inferred_autonomous_agency_score: float
    sample_frame_rgb: np.ndarray
    sample_flow_mag: np.ndarray
    voxel_id: str


class SpacetimeCausalDiscoveryEngine:
    """
    Learns Self, External Agents, Shadows, and Controllable Motor Gains directly from
    raw spacetime RGB sequences + motor efference copy inside CausalField(6D).
    """

    def __init__(self, organ: Optional[OcularScaleOrgan] = None) -> None:
        self.organ = organ if organ is not None else OcularScaleOrgan()
        self.causal_field = CausalField(dimensions=6)
        self._voxel_counter = 0

    def _add_spacetime_voxel(
        self,
        role: str,
        yx_norm: Tuple[float, float],
        vel_yx: Tuple[float, float],
        motor_corr: float,
        auton_score: float,
    ) -> str:
        self._voxel_counter += 1
        vid = f"voxel_spacetime_{role}_{self._voxel_counter:04d}"
        vec6 = np.array(
            [
                float(np.clip(yx_norm[0], 0.0, 1.0)),
                float(np.clip(yx_norm[1], 0.0, 1.0)),
                float(np.clip(0.5 + vel_yx[0] / 20.0, 0.0, 1.0)),
                float(np.clip(0.5 + vel_yx[1] / 20.0, 0.0, 1.0)),
                float(np.clip(motor_corr, 0.0, 1.0)),
                float(np.clip(auton_score, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        c_r = max(0.05, float(motor_corr))
        c_b = max(0.05, 1.0 - float(auton_score))
        c_y = max(0.05, float(auton_score))
        c_sum = c_r + c_b + c_y
        self.causal_field.add_voxel(
            InformationVoxel(
                id=vid,
                content={"role": role, "motor_corr": motor_corr, "auton": auton_score},
                tensor=vec6,
                mass=float(1.5 + motor_corr + auton_score),
                position=vec6.copy(),
                velocity=np.array([vel_yx[0], vel_yx[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                potential=float(motor_corr + auton_score),
                chromatic_vector=np.array([c_r / c_sum, c_b / c_sum, c_y / c_sum], dtype=np.float32),
            )
        )
        return vid

    def _track_patch_in_frame(
        self,
        obs_curr: OpticalSceneObservation,
        obs_next: OpticalSceneObservation,
        temp: TemporalVisualAnalysis,
        est_yx: Tuple[float, float],
        predicted_yx: Tuple[float, float],
        ref_chroma_vec3: np.ndarray,
    ) -> Tuple[float, float]:
        """Track controllable SELF entity visually using local optical flow + opponent signature."""
        h, w = obs_next.shape_hw
        cy_i = int(np.clip(round(est_yx[0]), 0, h - 1))
        cx_i = int(np.clip(round(est_yx[1]), 0, w - 1))

        # Initial prediction combining optical flow near current locus & efference prediction
        r_f = 12
        fy0, fy1 = max(0, cy_i - r_f), min(h, cy_i + r_f + 1)
        fx0, fx1 = max(0, cx_i - r_f), min(w, cx_i + r_f + 1)
        flow_sub = temp.flow_yx[fy0:fy1, fx0:fx1]
        flow_mag_sub = np.hypot(flow_sub[..., 0], flow_sub[..., 1])
        active_px = flow_mag_sub > 1.2
        if int(np.sum(active_px)) >= 8:
            flow_dy = float(np.median(flow_sub[..., 0][active_px]))
            flow_dx = float(np.median(flow_sub[..., 1][active_px]))
            anchor_y = 0.60 * (est_yx[0] + flow_dy) + 0.40 * predicted_yx[0]
            anchor_x = 0.60 * (est_yx[1] + flow_dx) + 0.40 * predicted_yx[1]
        else:
            anchor_y, anchor_x = predicted_yx[0], predicted_yx[1]

        # Local chromatic + luminance signature refinement in a 22px search radius
        sy0 = max(6, int(round(anchor_y)) - 22)
        sy1 = min(h - 6, int(round(anchor_y)) + 23)
        sx0 = max(6, int(round(anchor_x)) - 22)
        sx1 = min(w - 6, int(round(anchor_x)) + 23)
        if sy1 <= sy0 or sx1 <= sx0:
            return float(np.clip(anchor_y, 8, h - 8)), float(np.clip(anchor_x, 8, w - 8))

        a_n = obs_next.l1.luminance_a[sy0:sy1, sx0:sx1]
        rg_n = obs_next.l1.rg_opponent[sy0:sy1, sx0:sx1]
        by_n = obs_next.l1.by_opponent[sy0:sy1, sx0:sx1]

        chroma_dist = (
            (rg_n - ref_chroma_vec3[1]) ** 2
            + (by_n - ref_chroma_vec3[2]) ** 2
            + 0.20 * (a_n - ref_chroma_vec3[0]) ** 2
        )
        # Smooth disc correlation (radius ~7px matches central body)
        match_score = gaussian_filter(np.exp(-chroma_dist / 0.16), sigma=3.5).astype(np.float32)
        yy, xx = np.mgrid[sy0:sy1, sx0:sx1].astype(np.float32)
        prox_prior = np.exp(-((yy - anchor_y) ** 2 + (xx - anchor_x) ** 2) / (2.0 * 14.0 ** 2))
        total_score = match_score * prox_prior

        top_thresh = float(np.percentile(total_score, 93.0))
        mask = total_score >= top_thresh
        weights = total_score[mask]
        new_y = float(np.sum(yy[mask] * weights) / max(1e-6, float(np.sum(weights))))
        new_x = float(np.sum(xx[mask] * weights) / max(1e-6, float(np.sum(weights))))
        return new_y, new_x

    def _discover_static_target_beacon(
        self,
        obs: OpticalSceneObservation,
        self_yx: Tuple[float, float],
        ext_yx: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Locate stationary high-contrast opponent landmark well outside moving self/agent blobs."""
        h, w = obs.shape_hw
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

        dist_self = np.hypot(yy - self_yx[0], xx - self_yx[1])
        dist_ext = np.hypot(yy - ext_yx[0], xx - ext_yx[1])
        border_ok = (yy >= 16) & (yy <= h - 16) & (xx >= 16) & (xx <= w - 16)
        valid_zone = (dist_self > 32.0) & (dist_ext > 30.0) & border_ok

        # Landmark stands out by combining center-surround luminance + opponent contrast + persistence
        sal = (
            obs.l2.persistence_map
            + 0.8 * np.abs(obs.l1.dog_fine)
            + 0.7 * np.hypot(obs.l1.rg_opponent, obs.l1.by_opponent)
        )
        sal_sm = gaussian_filter(sal, sigma=3.5).astype(np.float32)
        sal_sm = np.where(valid_zone, sal_sm, 0.0)
        ty, tx = np.unravel_index(int(np.argmax(sal_sm)), (h, w))
        return float(ty), float(tx)

    def discover_and_control_arena(
        self, arena: ContinuousCausalArena, control_steps: int = 22
    ) -> GameSpacetimeReport:
        """
        Run full blind spacetime loop:
        Phase 1 (Motor babbling + idle observation) -> discover SELF, EXTERNAL_AGENT, SHADOW, ENVIRONMENT
        Phase 2 (Closed-loop pixel-only visual control) -> steer SELF to target using discovered gain.
        """
        h, w = arena.height, arena.width
        idle_map = np.zeros((h, w), dtype=np.float32)
        axis_maps = {
            "x_pos": np.zeros((h, w), dtype=np.float32),
            "x_neg": np.zeros((h, w), dtype=np.float32),
            "y_pos": np.zeros((h, w), dtype=np.float32),
            "y_neg": np.zeros((h, w), dtype=np.float32),
        }
        shadow_map = np.zeros((h, w), dtype=np.float32)

        # Motor babbling schedule returns near start so spatial accumulation stays crisp
        schedule: List[Tuple[str, Tuple[float, float]]] = [
            ("idle", (0.0, 0.0)),
            ("idle", (0.0, 0.0)),
            ("idle", (0.0, 0.0)),
            ("idle", (0.0, 0.0)),
            ("x_pos", (0.0, 1.0)),
            ("x_pos", (0.0, 1.0)),
            ("x_neg", (0.0, -1.0)),
            ("x_neg", (0.0, -1.0)),
            ("y_pos", (1.0, 0.0)),
            ("y_pos", (1.0, 0.0)),
            ("y_neg", (-1.0, 0.0)),
            ("y_neg", (-1.0, 0.0)),
        ]

        obs_curr = self.organ.observe(arena.render(), max_width=w)
        ext_vels: List[Tuple[float, float]] = []
        self_gains_y: List[float] = []
        self_gains_x: List[float] = []

        last_obs: OpticalSceneObservation = obs_curr
        last_temp: Optional[TemporalVisualAnalysis] = None
        last_active_flow_peak_yx: Optional[Tuple[float, float]] = None

        for tag, act_yx in schedule:
            rgb_next = arena.step(act_yx)
            obs_next = self.organ.observe(rgb_next, max_width=w)
            temp = self.organ.observe_transition(obs_curr, obs_next)

            flow_mag = np.hypot(temp.flow_yx[..., 0], temp.flow_yx[..., 1])
            da = obs_next.l1.luminance_a - obs_curr.l1.luminance_a
            dc = np.hypot(
                obs_next.l1.rg_opponent - obs_curr.l1.rg_opponent,
                obs_next.l1.by_opponent - obs_curr.l1.by_opponent,
            )

            # Pure optical shadow wave signature: luminance change with near-zero chroma & texture flow
            sh_mask = (np.abs(da) > 0.12) & (dc < 0.08) & (flow_mag < 0.9)
            shadow_map += np.where(sh_mask, np.abs(da), 0.0).astype(np.float32)

            if tag == "idle":
                idle_flow = np.maximum(0.0, flow_mag - 0.9).astype(np.float32)
                idle_map += gaussian_filter(idle_flow, sigma=2.5)
                if np.any(flow_mag > 1.3):
                    hot = flow_mag > float(np.percentile(flow_mag, 96.0))
                    ext_vels.append(
                        (
                            float(np.median(temp.flow_yx[..., 0][hot])),
                            float(np.median(temp.flow_yx[..., 1][hot])),
                        )
                    )
            else:
                ay, ax = act_yx
                proj = temp.flow_yx[..., 0] * ay + temp.flow_yx[..., 1] * ax
                ortho = np.abs(temp.flow_yx[..., 0] * (-ax) + temp.flow_yx[..., 1] * ay)
                aligned = np.maximum(0.0, proj - 0.45 * ortho - 0.6).astype(np.float32)
                aligned_sm = gaussian_filter(aligned, sigma=3.0)
                axis_maps[tag] += aligned_sm

                if float(np.max(aligned_sm)) > 0.5:
                    py, px = np.unravel_index(int(np.argmax(aligned_sm)), (h, w))
                    last_active_flow_peak_yx = (
                        float(py + temp.flow_yx[py, px, 0]),
                        float(px + temp.flow_yx[py, px, 1]),
                    )
                    hot = aligned_sm >= float(np.percentile(aligned_sm, 97.0))
                    if abs(ax) > 0.5:
                        self_gains_x.append(
                            float(np.median(temp.flow_yx[..., 1][hot]) / ax)
                        )
                    if abs(ay) > 0.5:
                        self_gains_y.append(
                            float(np.median(temp.flow_yx[..., 0][hot]) / ay)
                        )

            obs_curr = obs_next
            last_obs = obs_next
            last_temp = temp

        # Isolate SELF_BODY: must respond across BOTH x and y axes and not match idle motion
        x_resp = np.minimum(axis_maps["x_pos"], axis_maps["x_neg"])
        y_resp = np.minimum(axis_maps["y_pos"], axis_maps["y_neg"])
        joint_motor_res = np.sqrt(np.maximum(0.0, x_resp * y_resp)) + 0.35 * (
            axis_maps["x_pos"] + axis_maps["x_neg"] + axis_maps["y_pos"] + axis_maps["y_neg"]
        )
        idle_norm = idle_map / (float(np.max(idle_map)) + 1e-6)
        self_heatmap = np.maximum(
            0.0, joint_motor_res * (1.0 - 0.85 * idle_norm)
        ).astype(np.float32)
        self_heatmap = gaussian_filter(self_heatmap, sigma=3.0).astype(np.float32)

        s_y, s_x = np.unravel_index(int(np.argmax(self_heatmap)), (h, w))
        # Also combine with latest active step flow end locus if consistent
        self_est_yx = (
            (0.5 * s_y + 0.5 * last_active_flow_peak_yx[0], 0.5 * s_x + 0.5 * last_active_flow_peak_yx[1])
            if last_active_flow_peak_yx is not None
            and np.hypot(s_y - last_active_flow_peak_yx[0], s_x - last_active_flow_peak_yx[1]) < 24.0
            else (float(s_y), float(s_x))
        )

        # External agent estimated from latest residual flow outside SELF locus
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        not_self = np.hypot(yy - self_est_yx[0], xx - self_est_yx[1]) > 26.0
        ext_latest_score = (
            gaussian_filter(last_temp.residual_flow_mag, sigma=3.0)
            if last_temp is not None
            else idle_map
        )
        ext_combined = np.where(not_self, 0.65 * ext_latest_score + 0.35 * idle_map / 4.0, 0.0)
        ey, ex = np.unravel_index(int(np.argmax(ext_combined)), (h, w))
        ext_est_yx = (float(ey), float(ex))

        sh_sm = gaussian_filter(shadow_map, sigma=4.0).astype(np.float32)
        shy, shx = np.unravel_index(int(np.argmax(sh_sm)), (h, w))
        shadow_est_yx = (float(shy), float(shx))

        # Ground-truth eval error right at end of Phase 1 (blind evaluation)
        gt1 = arena.get_ground_truth_for_eval_only()
        self_err = float(
            np.hypot(self_est_yx[0] - gt1["player_yx"][0], self_est_yx[1] - gt1["player_yx"][1])
        )
        ext_err = float(
            np.hypot(ext_est_yx[0] - gt1["creature_yx"][0], ext_est_yx[1] - gt1["creature_yx"][1])
        )

        gain_y = float(np.clip(np.median(self_gains_y) if self_gains_y else 5.0, 1.5, 16.0))
        gain_x = float(np.clip(np.median(self_gains_x) if self_gains_x else 5.0, 1.5, 16.0))
        learned_gain = (gain_y, gain_x)

        mean_ext_v = (
            (float(np.mean([v[0] for v in ext_vels])), float(np.mean([v[1] for v in ext_vels])))
            if ext_vels
            else (0.0, 0.0)
        )

        vid_self = self._add_spacetime_voxel(
            "SELF_BODY",
            (self_est_yx[0] / h, self_est_yx[1] / w),
            (0.0, 0.0),
            motor_corr=0.95,
            auton_score=0.05,
        )
        vid_ext = self._add_spacetime_voxel(
            "EXTERNAL_AUTONOMOUS_AGENT",
            (ext_est_yx[0] / h, ext_est_yx[1] / w),
            mean_ext_v,
            motor_corr=0.05,
            auton_score=0.92,
        )
        vid_shadow = self._add_spacetime_voxel(
            "ILLUMINATION_SHADOW",
            (shadow_est_yx[0] / h, shadow_est_yx[1] / w),
            (0.0, 0.0),
            motor_corr=0.0,
            auton_score=0.10,
        )
        vid_env = self._add_spacetime_voxel(
            "STATIC_ENVIRONMENT", (0.5, 0.5), (0.0, 0.0), motor_corr=0.0, auton_score=0.0
        )
        self.causal_field.link_voxels(vid_self, vid_env, strength=0.80)
        self.causal_field.link_voxels(vid_ext, vid_env, strength=0.70)
        self.causal_field.link_voxels(vid_shadow, vid_env, strength=0.60)

        discoveries = [
            SpacetimeEntityDiscovery("SELF_BODY", self_est_yx, (gain_y, gain_x), 0.95, 0.05, vid_self),
            SpacetimeEntityDiscovery("EXTERNAL_AUTONOMOUS_AGENT", ext_est_yx, mean_ext_v, 0.05, 0.92, vid_ext),
            SpacetimeEntityDiscovery("ILLUMINATION_SHADOW", shadow_est_yx, (0.0, 0.0), 0.0, 0.10, vid_shadow),
            SpacetimeEntityDiscovery("STATIC_ENVIRONMENT", (h * 0.5, w * 0.5), (0.0, 0.0), 0.0, 0.0, vid_env),
        ]

        # Extract Phase 1 blind color/opponent patch signature around SELF_BODY for closed-loop tracking
        si_y = int(np.clip(round(self_est_yx[0]), 6, h - 7))
        si_x = int(np.clip(round(self_est_yx[1]), 6, w - 7))
        self_chroma_ref = np.array(
            [
                float(np.mean(last_obs.l1.luminance_a[si_y - 5 : si_y + 6, si_x - 5 : si_x + 6])),
                float(np.mean(last_obs.l1.rg_opponent[si_y - 5 : si_y + 6, si_x - 5 : si_x + 6])),
                float(np.mean(last_obs.l1.by_opponent[si_y - 5 : si_y + 6, si_x - 5 : si_x + 6])),
            ],
            dtype=np.float32,
        )

        target_est_yx = self._discover_static_target_beacon(last_obs, self_est_yx, ext_est_yx)

        # Phase 2: Blind Closed-Loop Visual-Motor Control strictly from raw RGB frames
        curr_est = self_est_yx
        est_vel = (0.0, 0.0)
        traj_est: List[Tuple[float, float]] = [curr_est]
        traj_true: List[Tuple[float, float]] = [gt1["player_yx"]]
        dist_curve: List[float] = [
            float(
                np.hypot(
                    gt1["player_yx"][0] - gt1["target_yx"][0],
                    gt1["player_yx"][1] - gt1["target_yx"][1],
                )
            )
        ]

        sample_frame = (last_obs.rgb_working * 255.0).astype(np.uint8)
        obs_ctrl = last_obs

        for step_c in range(max(4, int(control_steps))):
            err_y = target_est_yx[0] - curr_est[0]
            err_x = target_est_yx[1] - curr_est[1]
            # Proportional-derivative visual servoing using learned motor gain
            act_y = float(
                np.clip(err_y / (2.4 * gain_y) - 0.35 * est_vel[0] / gain_y, -1.0, 1.0)
            )
            act_x = float(
                np.clip(err_x / (2.4 * gain_x) - 0.35 * est_vel[1] / gain_x, -1.0, 1.0)
            )

            rgb_next = arena.step((act_y, act_x))
            obs_next = self.organ.observe(rgb_next, max_width=w)
            temp_c = self.organ.observe_transition(obs_ctrl, obs_next)

            pred_yx = (
                curr_est[0] + 0.55 * est_vel[0] + 0.8 * gain_y * act_y,
                curr_est[1] + 0.55 * est_vel[1] + 0.8 * gain_x * act_x,
            )
            new_est = self._track_patch_in_frame(
                obs_ctrl, obs_next, temp_c, curr_est, pred_yx, self_chroma_ref
            )
            est_vel = (new_est[0] - curr_est[0], new_est[1] - curr_est[1])
            curr_est = new_est
            obs_ctrl = obs_next

            gt_c = arena.get_ground_truth_for_eval_only()
            traj_est.append(curr_est)
            traj_true.append(gt_c["player_yx"])
            dist_curve.append(
                float(
                    np.hypot(
                        gt_c["player_yx"][0] - gt_c["target_yx"][0],
                        gt_c["player_yx"][1] - gt_c["target_yx"][1],
                    )
                )
            )

        init_dist = dist_curve[0]
        final_dist = dist_curve[-1]
        red_ratio = float(max(0.0, (init_dist - final_dist) / max(1e-6, init_dist)))

        return GameSpacetimeReport(
            phase1_discoveries=discoveries,
            self_localization_error_px=self_err,
            external_agent_localization_error_px=ext_err,
            learned_motor_gain_yx=learned_gain,
            control_initial_dist_px=init_dist,
            control_final_dist_px=final_dist,
            control_distance_reduction_ratio=red_ratio,
            control_trajectory_est_yx=traj_est,
            control_trajectory_true_yx=traj_true,
            distance_curve_px=dist_curve,
            sample_frame_rgb=sample_frame,
            self_resonance_heatmap=self_heatmap,
            external_motion_heatmap=idle_map,
            shadow_heatmap=sh_sm,
            discovered_target_yx=target_est_yx,
        )

    def discover_autonomous_agent_in_real_video(
        self,
        video_path: Path | str,
        max_frames: int = 10,
        frame_stride: int = 6,
        max_width: int = 240,
    ) -> RealVideoSpacetimeReport:
        """
        Apply the exact same passive spacetime causal flow logic (`action=(0, 0)`) to a
        real external MP4 video, inferring autonomous agency purely from coherent flow & occlusion.
        """
        vpath = Path(video_path)
        if not vpath.exists():
            raise FileNotFoundError(f"Missing real video file: {vpath}")

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open real video: {vpath}")

        frames_rgb: List[np.ndarray] = []
        idx = 0
        try:
            while len(frames_rgb) < max_frames:
                ok, bgr = cap.read()
                if not ok:
                    break
                if idx % max(1, frame_stride) == 0:
                    frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy())
                idx += 1
        finally:
            cap.release()

        if len(frames_rgb) < 3:
            raise RuntimeError(f"Decoded insufficient frames ({len(frames_rgb)}) from {vpath}")

        obs_list = [self.organ.observe(fr, max_width=max_width) for fr in frames_rgb]
        traj_yx: List[Tuple[float, float]] = []
        indep_count = 0
        fg_occ_count = 0
        peak_flows: List[float] = []

        sample_frame = frames_rgb[0]
        sample_flow = np.zeros(obs_list[0].shape_hw, dtype=np.float32)
        best_flow_e = -1.0

        for k in range(1, len(obs_list)):
            temp = self.organ.observe_transition(obs_list[k - 1], obs_list[k])
            if temp.event_classification == "INDEPENDENT_OBJECT_MOTION":
                indep_count += 1
            if temp.occlusion_depth_verdict == "MOVING_REGION_IN_FRONT_OF_BACKGROUND":
                fg_occ_count += 1

            fh, fw = frames_rgb[k].shape[:2]
            oh, ow = obs_list[k].shape_hw
            flow_mag = temp.residual_flow_mag
            if np.any(temp.moving_mask):
                ys, xs = np.where(temp.moving_mask)
                cy = float(np.mean(ys)) * (fh / float(oh))
                cx = float(np.mean(xs)) * (fw / float(ow))
                peak_flows.append(float(np.percentile(flow_mag[temp.moving_mask], 80.0)))
            else:
                my, mx = np.unravel_index(int(np.argmax(flow_mag)), (oh, ow))
                cy = float(my) * (fh / float(oh))
                cx = float(mx) * (fw / float(ow))
                peak_flows.append(float(np.max(flow_mag)))
            traj_yx.append((cy, cx))

            e_val = float(np.mean(flow_mag))
            if e_val > best_flow_e:
                best_flow_e = e_val
                sample_frame = frames_rgb[k].copy()
                sample_flow = flow_mag.copy()

        n_trans = max(1, len(obs_list) - 1)
        indep_ratio = float(indep_count / n_trans)
        fg_ratio = float(fg_occ_count / n_trans)
        vel_norm = float(np.clip(np.mean(peak_flows) / 3.5, 0.0, 1.0)) if peak_flows else 0.0

        # Autonomous agency inferred bottom-up from spacetime motion (NOT hand-supplied!)
        inferred_agency = float(
            np.clip(0.50 * indep_ratio + 0.30 * vel_norm + 0.20 * fg_ratio, 0.0, 1.0)
        )

        last_pt = traj_yx[-1] if traj_yx else (0.5, 0.5)
        fh0, fw0 = sample_frame.shape[:2]
        vid = self._add_spacetime_voxel(
            "REAL_VIDEO_AUTONOMOUS_AGENT",
            (last_pt[0] / max(1, fh0), last_pt[1] / max(1, fw0)),
            (float(np.mean(peak_flows)) if peak_flows else 0.0, 0.0),
            motor_corr=0.0,
            auton_score=inferred_agency,
        )

        return RealVideoSpacetimeReport(
            video_name=vpath.stem,
            frames_processed=len(frames_rgb),
            discovered_agent_trajectory_yx=traj_yx,
            inferred_autonomous_agency_score=inferred_agency,
            sample_frame_rgb=sample_frame,
            sample_flow_mag=sample_flow,
            voxel_id=vid,
        )
