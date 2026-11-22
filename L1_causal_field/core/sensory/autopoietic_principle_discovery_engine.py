r"""
autopoietic_principle_discovery_engine.py

Autopoietic Structural Principle Discovery & Self-Synthesis Engine
==================================================================
Embodies the AGENTS.md principles of Continuous Causal Intelligence, Inverse Mechanism
Generation, and Epistemic Humility:

Instead of hand-coding separate hardcoded modules for AAA game VFX clutter,
hidden-state occlusion physics, or black-box engine laws, ELYSIA exposes a single
unified autonomous discovery loop (`confront_unknown_world`) that:
1. Confronts an unlabeled blind world through a baseline 0-order instantaneous lens
   and measures empirical prediction residuals (friction at the sensory boundary).
2. Evaluates an MDL-ordered ladder of structural organ hypotheses by generating
   executable Python DSP / dynamical-law source code parameterized solely from
   training telemetry, validating AST safety and compiling inside `SafeASTSandbox`.
3. Tests every hypothesis on **unseen** trajectories (out-of-sample generalization &
   unexplained variance check) so wrong or overfit hypotheses are falsified rather
   than hallucinated.
4. Hot-swaps only a genuinely unifying causal principle into `CausalField`
   (`InformationVoxel` + `ConnectivityBeam`) and `AgamottoCausalRotorEngramEngine`,
   while explicitly invoking **Epistemic Boundary Refusal** on irreducible noise.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from core.physics.causal_field import CausalField, InformationVoxel
from core.sensory.agamotto_causal_rotor_engram_engine import (
    AgamottoCausalRotorEngramEngine,
)
from core.sensory.autopoietic_organ_genesis import SafeASTSandbox


DEFAULT_PRINCIPLE_DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_autopoietic_principle_discovery_dashboard.png"
)


# ======================================================================
# 1. Domain-Agnostic Telemetry & Self-Discovery Data Contracts
# ======================================================================
@dataclass
class WorldTelemetrySplit:
    """Generic unlabeled physical telemetry presented to ELYSIA."""

    targets: np.ndarray
    frames: Optional[np.ndarray] = None
    observed_series: Optional[np.ndarray] = None
    visible_mask: Optional[np.ndarray] = None
    actions: Optional[np.ndarray] = None
    state_inputs: Optional[np.ndarray] = None


@dataclass
class ResidualDiagnosis:
    mse_error: float
    residual_autocorr_lag1: float
    spatial_clutter_ratio: float
    causal_signal_snr: float
    diagnosis_tag: str


@dataclass
class HypothesisAttempt:
    organ_name: str
    ast_nodes: int
    train_error: float
    unseen_error: float
    mdl_score: float
    status: str  # "BASELINE" | "ACCEPTED_MOUNTED" | "FALSIFIED_ERROR" | "FALSIFIED_MDL"


@dataclass
class WorldDiscoveryResult:
    world_id: str
    baseline_error: float
    final_error: float
    error_reduction_ratio: float
    discovered_organ_name: Optional[str]
    discovered_source_code: Optional[str]
    ast_nodes: int
    attempts: List[HypothesisAttempt]
    diagnosis: ResidualDiagnosis
    epistemic_status: str  # "PRINCIPLE_DISCOVERED_AND_MOUNTED" | "EPISTEMIC_BOUNDARY_IRREDUCIBLE_NOISE"
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    _unseen_targets: Optional[np.ndarray] = field(default=None, repr=False)
    _baseline_preds: Optional[np.ndarray] = field(default=None, repr=False)
    _final_preds: Optional[np.ndarray] = field(default=None, repr=False)
    _observed_series: Optional[np.ndarray] = field(default=None, repr=False)
    _visible_mask: Optional[np.ndarray] = field(default=None, repr=False)


# ======================================================================
# 2. Four Blind External Worlds (Pure Generative Physics, Zero Hints)
# ======================================================================
class ClutteredVFXActionWorld:
    """
    Trial 1: Mimics visual perception inside an action scene / AAA game with
    a bright drifting background gradient, intense random flashing VFX sparks,
    and a smaller localized avatar continuously coupled to motor `actions`.
    """

    def __init__(self, height: int = 24, width: int = 24) -> None:
        self.height = height
        self.width = width

    def rollout(self, length: int = 96, seed: int = 101) -> WorldTelemetrySplit:
        rng = np.random.default_rng(seed)
        t_idx = np.arange(length, dtype=np.float64)
        # Non-periodic multi-frequency motor command sequence
        actions = (
            0.75 * np.sin(0.27 * t_idx + rng.uniform(0.0, 2.0 * np.pi))
            + 0.45 * np.cos(0.61 * t_idx + rng.uniform(0.0, 2.0 * np.pi))
            + 0.35 * rng.normal(0.0, 1.0, size=length)
        )
        actions = np.clip(actions, -1.5, 1.5)

        obj_x = 0.45 + 0.08 * rng.normal()
        obj_y = 0.50
        vel_x = 0.0
        true_x = np.zeros(length, dtype=np.float64)

        xs = np.linspace(0.0, 1.0, self.width, dtype=np.float64)[None, :]
        ys = np.linspace(0.0, 1.0, self.height, dtype=np.float64)[:, None]
        frames = np.zeros((length, self.height, self.width), dtype=np.float64)

        bg_phase = rng.uniform(0.0, 2.0 * np.pi)
        for t in range(length):
            vel_x = 0.72 * vel_x + 0.055 * actions[t] - 0.035 * (obj_x - 0.50)
            obj_x = float(np.clip(obj_x + vel_x, 0.14, 0.86))
            true_x[t] = obj_x

            # 1) Bright drifting low-frequency background wave across entire viewport
            bg = 0.34 + 0.24 * np.cos(2.0 * np.pi * xs - 0.09 * t + bg_phase) * (
                0.70 + 0.30 * np.sin(np.pi * ys)
            )
            # 2) Intense high-energy random VFX sparks uncorrelated with action
            sparks = np.zeros((self.height, self.width), dtype=np.float64)
            n_sparks = int(rng.integers(3, 6))
            for _ in range(n_sparks):
                sx = float(rng.uniform(0.06, 0.94))
                sy = float(rng.uniform(0.06, 0.94))
                amp = float(rng.uniform(1.15, 1.75))
                sparks += amp * np.exp(-(((xs - sx) ** 2 + (ys - sy) ** 2) / (2.0 * 0.055**2)))

            # 3) Action-coupled foreground avatar
            avatar = 0.95 * np.exp(-(((xs - obj_x) ** 2 + (ys - obj_y) ** 2) / (2.0 * 0.065**2)))
            frames[t] = bg + sparks + avatar

        return WorldTelemetrySplit(
            frames=frames.astype(np.float32),
            actions=actions.astype(np.float32),
            targets=true_x.astype(np.float64),
        )


class OccludedMomentumWorld:
    """
    Trial 2: Harmonic oscillator hidden behind an opaque screen whenever x >= 0.44.
    Inside the blind tunnel the object reaches its turning apex and rebounds.
    A memoryless lens freezes, linear constant velocity overshoots out of bounds;
    only a synthesized 2nd-order inertial phase memory tracks across occlusion.
    """

    def __init__(self, damping: float = 0.992, stiffness: float = 0.048, eq: float = 0.50) -> None:
        self.damping = damping
        self.stiffness = stiffness
        self.eq = eq

    def rollout(
        self, length: int = 120, x0: float = 0.14, v0: float = 0.032, occlude_threshold: float = 0.44
    ) -> WorldTelemetrySplit:
        x = float(x0)
        v = float(v0)
        true_x = np.zeros(length, dtype=np.float64)
        for t in range(length):
            true_x[t] = x
            v = self.damping * v - self.stiffness * (x - self.eq)
            x = x + v

        visible_mask = true_x < occlude_threshold
        observed = np.where(visible_mask, true_x, np.nan)
        return WorldTelemetrySplit(
            observed_series=observed.astype(np.float64),
            visible_mask=visible_mask.astype(bool),
            targets=true_x.astype(np.float64),
        )


class OpaqueEngineTickWorld:
    """
    Trial 3: Black-box simulation engine `tick(pos, vel, action)` governed by
    hidden internal equations. ELYSIA only observes probe trajectories and must
    invert the mathematical transition law as executable Python code.
    """

    def tick(self, pos: float, vel: float, action: float) -> Tuple[float, float]:
        next_vel = 0.88 * vel - 0.14 * pos + 0.35 * action
        next_pos = pos + next_vel
        return float(next_pos), float(next_vel)

    def rollout(self, length: int = 80, seed: int = 301, action_scale: float = 1.0) -> WorldTelemetrySplit:
        rng = np.random.default_rng(seed)
        pos = float(rng.uniform(-0.8, 0.8))
        vel = float(rng.uniform(-0.3, 0.3))
        inputs = np.zeros((length, 3), dtype=np.float64)
        targets = np.zeros((length, 2), dtype=np.float64)

        for t in range(length):
            act = float(
                action_scale
                * (
                    0.8 * np.sin(0.31 * t + seed * 0.1)
                    + 0.5 * rng.normal()
                )
            )
            npos, nvel = self.tick(pos, vel, act)
            inputs[t] = [pos, vel, act]
            targets[t] = [npos, nvel]
            pos, vel = npos, nvel

        return WorldTelemetrySplit(state_inputs=inputs, targets=targets)


class IrreducibleNoiseWorld:
    """
    Trial 4: Pure i.i.d. uniform white noise with partial random masking.
    Zero temporal or causal structure exists; any complex predictor overfits.
    """

    def rollout(self, length: int = 120, seed: int = 404) -> WorldTelemetrySplit:
        rng = np.random.default_rng(seed)
        true_x = rng.uniform(0.15, 0.85, size=length)
        visible_mask = rng.random(size=length) > 0.35
        observed = np.where(visible_mask, true_x, np.nan)
        return WorldTelemetrySplit(
            observed_series=observed.astype(np.float64),
            visible_mask=visible_mask.astype(bool),
            targets=true_x.astype(np.float64),
        )


class CompositeVFXOccludedWorld:
    """
    Trial 5: Composite blind world combining bright drifting background, random
    flashing VFX sparks, AND complete visual occlusion whenever x >= occlude_threshold.
    Requires autonomously recombining perceptual separation + latent momentum memory.
    """

    def __init__(
        self,
        height: int = 24,
        width: int = 24,
        damping: float = 0.992,
        stiffness: float = 0.048,
        eq: float = 0.50,
    ) -> None:
        self.height = height
        self.width = width
        self.damping = damping
        self.stiffness = stiffness
        self.eq = eq

    def rollout(
        self,
        length: int = 120,
        x0: float = 0.14,
        v0: float = 0.032,
        occlude_threshold: float = 0.46,
        seed: int = 505,
        expose_mask_for_eval: bool = False,
    ) -> WorldTelemetrySplit:
        rng = np.random.default_rng(seed)
        x = float(x0)
        v = float(v0)
        true_x = np.zeros(length, dtype=np.float64)
        xs = np.linspace(0.0, 1.0, self.width, dtype=np.float64)[None, :]
        ys = np.linspace(0.0, 1.0, self.height, dtype=np.float64)[:, None]
        frames = np.zeros((length, self.height, self.width), dtype=np.float64)
        bg_phase = rng.uniform(0.0, 2.0 * np.pi)

        for t in range(length):
            true_x[t] = x
            bg = 0.34 + 0.24 * np.cos(2.0 * np.pi * xs - 0.09 * t + bg_phase) * (
                0.70 + 0.30 * np.sin(np.pi * ys)
            )
            sparks = np.zeros((self.height, self.width), dtype=np.float64)
            n_sparks = int(rng.integers(3, 6))
            for _ in range(n_sparks):
                sx = float(rng.uniform(0.06, 0.94))
                sy = float(rng.uniform(0.06, 0.94))
                amp = float(rng.uniform(1.10, 1.70))
                sparks += amp * np.exp(-(((xs - sx) ** 2 + (ys - sy) ** 2) / (2.0 * 0.055**2)))

            if x < occlude_threshold:
                avatar = 1.05 * np.exp(-(((xs - x) ** 2 + (ys - 0.50) ** 2) / (2.0 * 0.065**2)))
            else:
                avatar = 0.0
            frames[t] = bg + sparks + avatar
            v = self.damping * v - self.stiffness * (x - self.eq)
            x = x + v

        vis_mask = (true_x < occlude_threshold).astype(bool) if expose_mask_for_eval else None
        return WorldTelemetrySplit(
            frames=frames.astype(np.float32),
            actions=np.zeros(length, dtype=np.float32),
            targets=true_x.astype(np.float64),
            visible_mask=vis_mask,
        )


class NonlinearDragEngineWorld:
    """
    Trial 6: Nonlinear black-box physics engine with quadratic aerodynamic drag
    `next_vel = 0.96 * vel - 0.12 * pos - 0.28 * vel * abs(vel) + 0.40 * act`.
    Linear inversion fails on high-speed out-of-distribution trajectories; requires
    template-free sparse discovery over atomic mathematical operators.
    """

    def tick(self, pos: float, vel: float, action: float) -> Tuple[float, float]:
        next_vel = 0.96 * vel - 0.12 * pos - 0.28 * vel * abs(vel) + 0.40 * action
        next_pos = pos + next_vel
        return float(next_pos), float(next_vel)

    def rollout(self, length: int = 100, seed: int = 601, action_scale: float = 1.0) -> WorldTelemetrySplit:
        rng = np.random.default_rng(seed)
        pos = float(rng.uniform(-0.8, 0.8))
        vel = float(rng.uniform(-0.4, 0.4))
        inputs = np.zeros((length, 3), dtype=np.float64)
        targets = np.zeros((length, 2), dtype=np.float64)
        for t in range(length):
            act = float(
                action_scale
                * (1.1 * np.sin(0.28 * t + seed * 0.1) + 0.65 * rng.normal())
            )
            npos, nvel = self.tick(pos, vel, act)
            inputs[t] = [pos, vel, act]
            targets[t] = [npos, nvel]
            pos, vel = npos, nvel
        return WorldTelemetrySplit(state_inputs=inputs, targets=targets)


# ======================================================================
# 3. Unified Autopoietic Principle Discovery & Synthesis Engine
# ======================================================================
class AutopoieticPrincipleDiscoveryEngine:
    """
    Single domain-agnostic self-discovery system anchored in `CausalField`.
    Identifies missing physical/cognitive principles from prediction residuals,
    synthesizes candidate Python organs, validates AST safety & execution in
    `SafeASTSandbox`, falsifies failed hypotheses via unseen error + MDL, and
    only mounts genuine laws.
    """

    def __init__(self) -> None:
        self.causal_field = CausalField(dimensions=6)
        self.rotor_engine = AgamottoCausalRotorEngramEngine()
        self.mounted_organs: Dict[str, str] = {}
        self.compiled_callables: Dict[str, Callable[..., np.ndarray]] = {}
        self.discovery_history: List[WorldDiscoveryResult] = []

    # ------------------------------------------------------------------
    # Source-Code Synthesis Templates (compiled inside SafeASTSandbox)
    # ------------------------------------------------------------------
    @staticmethod
    def _src_l0_global_instant_centroid() -> str:
        return (
            "def predict(frames, actions):\n"
            "    w = frames.shape[2]\n"
            "    xs = np.linspace(0.0, 1.0, w, dtype=np.float64)\n"
            "    weights = np.maximum(frames.astype(np.float64), 0.0)\n"
            "    row_sum = np.sum(weights, axis=1)\n"
            "    total = np.sum(row_sum, axis=1) + 1e-12\n"
            "    return np.sum(row_sum * xs[None, :], axis=1) / total\n"
        )

    @staticmethod
    def _src_l1_temporal_diff_filter() -> str:
        return (
            "def predict(frames, actions):\n"
            "    t_len = frames.shape[0]\n"
            "    w = frames.shape[2]\n"
            "    xs = np.linspace(0.0, 1.0, w, dtype=np.float64)\n"
            "    f64 = frames.astype(np.float64)\n"
            "    out = np.zeros(t_len, dtype=np.float64)\n"
            "    w0 = np.maximum(f64[0], 0.0)\n"
            "    out[0] = float(np.sum(np.sum(w0, axis=0) * xs) / (np.sum(w0) + 1e-12))\n"
            "    for t in range(1, t_len):\n"
            "        diff_map = np.abs(f64[t] - f64[t - 1])\n"
            "        col_energy = np.sum(diff_map, axis=0)\n"
            "        tot = float(np.sum(col_energy)) + 1e-12\n"
            "        out[t] = float(np.sum(col_energy * xs) / tot)\n"
            "    return out\n"
        )

    @staticmethod
    def _synthesize_l2_action_coherence_organ(train: WorldTelemetrySplit) -> str:
        """
        Infers which spatial bands / pixels coherently respond to `actions`
        using optical-flow-action correlation `(dI/dt * dI/dx) * action[t]` vs
        incoherent transient pixel flashing variance, and compiles a standalone
        Python attention-coupled perceptual tracking organ.
        """
        assert train.frames is not None and train.actions is not None
        f64 = np.maximum(np.asarray(train.frames, dtype=np.float64), 0.0)
        act = np.asarray(train.actions, dtype=np.float64)
        t_tr, h, w = f64.shape
        xs_grid = np.linspace(0.0, 1.0, w, dtype=np.float64)

        # 1) Spatial Center-Surround (DoG) strips slow drifting wide background
        s3 = (f64 + np.roll(f64, 1, 2) + np.roll(f64, -1, 2) + np.roll(f64, 1, 1) + np.roll(f64, -1, 1)) / 5.0
        s7 = (
            s3
            + np.roll(s3, 2, 2)
            + np.roll(s3, -2, 2)
            + np.roll(s3, 2, 1)
            + np.roll(s3, -2, 1)
            + np.roll(s3, 4, 2)
            + np.roll(s3, -4, 2)
        ) / 7.0
        dog = np.maximum(f64 - 0.95 * s7, 0.0)

        # 2) Temporal Persistence product crushes 1-frame random flashing VFX sparks
        pers_3d = np.zeros_like(dog)
        for ti in range(t_tr):
            ref = dog[1] if (ti == 0 and t_tr > 1) else dog[max(0, ti - 1)]
            ref_blur = (
                0.20 * np.roll(ref, -2, 1)
                + 0.20 * np.roll(ref, -1, 1)
                + 0.20 * ref
                + 0.20 * np.roll(ref, 1, 1)
                + 0.20 * np.roll(ref, 2, 1)
            )
            pers_3d[ti] = np.sqrt(np.maximum(dog[ti] * ref_blur, 0.0))

        # 3) Row salience combined with action-motion coherence (zero prior target geometry)
        pr_energy = np.mean(np.max(pers_3d[1:], axis=2), axis=0)
        row_sums = np.sum(pers_3d, axis=2) + 1e-9
        row_cx = np.sum(pers_3d * xs_grid[None, None, :], axis=2) / row_sums  # (T, H)
        row_dx = np.diff(row_cx, axis=0)
        act_c = act[:-1] - float(np.mean(act[:-1]))
        row_act_cov = np.abs(np.mean(row_dx * act_c[:, None], axis=0))
        row_salience = np.maximum(pr_energy - float(np.percentile(pr_energy, 55)), 0.0) * (
            1.0 + 10.0 * (row_act_cov / (float(np.max(row_act_cov)) + 1e-9))
        )
        if float(np.max(row_salience)) < 1e-12:
            row_salience = pr_energy + 1e-6
        row_weights = np.power(row_salience / (float(np.max(row_salience)) + 1e-12), 2.5)
        row_weights = row_weights / (float(np.sum(row_weights)) + 1e-12)

        # 4) Extract coarse continuous trajectory and fit dynamical law via robust IRLS
        band_seq = np.sum(pers_3d * row_weights[None, :, None], axis=1)
        raw_x = np.zeros(t_tr, dtype=np.float64)
        cur_p = float(xs_grid[int(np.argmax(band_seq[0]))])
        raw_x[0] = cur_p
        for t in range(1, t_tr):
            focus = np.exp(-0.5 * ((xs_grid - cur_p) / 0.16) ** 2)
            bw = band_seq[t] * focus
            bs = float(np.sum(bw))
            if bs > 1e-6:
                cur_p = float(np.sum(bw * xs_grid) / bs)
            else:
                cur_p = float(xs_grid[int(np.argmax(band_seq[t]))])
            raw_x[t] = cur_p

        if len(raw_x) >= 12:
            dxs = np.diff(raw_x)
            X_tr = np.column_stack(
                [dxs[:-1], act[1:-1], raw_x[1:-1], np.ones(len(dxs) - 1, dtype=np.float64)]
            )
            y_tr = dxs[1:]
            mask_in = np.ones(len(y_tr), dtype=bool)
            coeffs = np.zeros(4, dtype=np.float64)
            for _ in range(3):
                reg = 1e-5 * np.eye(4, dtype=np.float64)
                Xm, ym = X_tr[mask_in], y_tr[mask_in]
                if len(ym) >= 6:
                    coeffs = np.linalg.solve(Xm.T @ Xm + reg, Xm.T @ ym)
                    res = np.abs(y_tr - X_tr @ coeffs)
                    thr = float(np.percentile(res, 82))
                    mask_in = res <= max(thr, 1e-6)
            k_v = float(np.clip(coeffs[0], -0.95, 0.95))
            k_a = float(np.clip(coeffs[1], -0.50, 0.50))
            k_x = float(np.clip(coeffs[2], -0.50, 0.50))
            k_0 = float(np.clip(coeffs[3], -0.50, 0.50))
        else:
            k_v, k_a, k_x, k_0 = 0.0, 0.0, 0.0, 0.0

        row_list = [round(float(v), 6) for v in row_weights.tolist()]

        return (
            "def predict(frames, actions):\n"
            f"    row_w = np.array({json.dumps(row_list)}, dtype=np.float64)\n"
            f"    k_v = {k_v:.6f}\n"
            f"    k_a = {k_a:.6f}\n"
            f"    k_x = {k_x:.6f}\n"
            f"    k_0 = {k_0:.6f}\n"
            "    t_len = frames.shape[0]\n"
            "    w = frames.shape[2]\n"
            "    xs = np.linspace(0.0, 1.0, w, dtype=np.float64)\n"
            "    f64 = np.maximum(frames.astype(np.float64), 0.0)\n"
            "    s3 = (f64 + np.roll(f64, 1, 2) + np.roll(f64, -1, 2) + np.roll(f64, 1, 1) + np.roll(f64, -1, 1)) / 5.0\n"
            "    s7 = (s3 + np.roll(s3, 2, 2) + np.roll(s3, -2, 2) + np.roll(s3, 2, 1) + np.roll(s3, -2, 1) + np.roll(s3, 4, 2) + np.roll(s3, -4, 2)) / 7.0\n"
            "    dog = np.maximum(f64 - 0.95 * s7, 0.0)\n"
            "    bseq = np.sum(dog * row_w[None, :, None], axis=1)\n"
            "    pers = np.zeros_like(bseq)\n"
            "    for ti in range(t_len):\n"
            "        ref = bseq[1] if (ti == 0 and t_len > 1) else bseq[max(0, ti - 1)]\n"
            "        ref_b = 0.20 * np.roll(ref, -2) + 0.20 * np.roll(ref, -1) + 0.20 * ref + 0.20 * np.roll(ref, 1) + 0.20 * np.roll(ref, 2)\n"
            "        pers[ti] = np.sqrt(np.maximum(bseq[ti] * ref_b, 0.0))\n"
            "    out = np.zeros(t_len, dtype=np.float64)\n"
            "    cur_x = float(xs[int(np.argmax(pers[0]))])\n"
            "    vel_est = 0.0\n"
            "    out[0] = cur_x\n"
            "    for t in range(1, t_len):\n"
            "        vel_est = k_v * vel_est + k_a * float(actions[t - 1]) + k_x * cur_x + k_0\n"
            "        pred_x = min(max(cur_x + vel_est, 0.08), 0.92)\n"
            "        focus = np.exp(-0.5 * ((xs - pred_x) / 0.16) ** 2)\n"
            "        band = pers[t] * focus\n"
            "        band_sum = float(np.sum(band))\n"
            "        if band_sum > 1e-6:\n"
            "            obs_x = float(np.sum(band * xs) / band_sum)\n"
            "            cur_x = 0.35 * pred_x + 0.65 * obs_x\n"
            "            vel_est = 0.45 * vel_est + 0.55 * (cur_x - out[t - 1])\n"
            "        else:\n"
            "            cur_x = pred_x\n"
            "        out[t] = cur_x\n"
            "    return out\n"
        )

    @staticmethod
    def _src_l0_memoryless_copy(train_mean: float) -> str:
        return (
            "def predict(observed_series, visible_mask):\n"
            "    n = len(observed_series)\n"
            "    out = np.zeros(n, dtype=np.float64)\n"
            f"    last_val = {train_mean:.6f}\n"
            "    for t in range(n):\n"
            "        out[t] = last_val\n"
            "        if visible_mask[t]:\n"
            "            last_val = float(observed_series[t])\n"
            "    return out\n"
        )

    @staticmethod
    def _src_l1_linear_extrapolation(train_mean: float) -> str:
        return (
            "def predict(observed_series, visible_mask):\n"
            "    n = len(observed_series)\n"
            "    out = np.zeros(n, dtype=np.float64)\n"
            f"    x_curr = {train_mean:.6f}\n"
            "    v_curr = 0.0\n"
            "    has_prev = False\n"
            "    prev_obs = x_curr\n"
            "    for t in range(n):\n"
            "        pred = x_curr + v_curr\n"
            "        out[t] = pred\n"
            "        if visible_mask[t]:\n"
            "            obs = float(observed_series[t])\n"
            "            if has_prev:\n"
            "                v_curr = obs - prev_obs\n"
            "            prev_obs = obs\n"
            "            x_curr = obs\n"
            "            has_prev = True\n"
            "        else:\n"
            "            x_curr = pred\n"
            "            has_prev = False\n"
            "    return out\n"
        )

    @staticmethod
    def _synthesize_l2_phase_delay_momentum_organ(train: WorldTelemetrySplit) -> str:
        """
        Extracts 2nd-order state recurrence `dx_{t} = c_v * dx_{t-1} + c_x * x_{t-1} + c_0`
        strictly from consecutive visible triplets on `train` (zero knowledge of occlusion
        bounds or true hidden states) and compiles a continuous latent oscillator tracker.
        """
        assert train.observed_series is not None and train.visible_mask is not None
        obs = np.asarray(train.observed_series, dtype=np.float64)
        vis = np.asarray(train.visible_mask, dtype=bool)
        valid_vals = obs[vis]
        train_mean = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.5

        rows_x: List[List[float]] = []
        rows_y: List[float] = []
        for t in range(2, len(obs)):
            if vis[t] and vis[t - 1] and vis[t - 2]:
                dx_prev = float(obs[t - 1] - obs[t - 2])
                dx_curr = float(obs[t] - obs[t - 1])
                rows_x.append([dx_prev, float(obs[t - 1]), 1.0])
                rows_y.append(dx_curr)

        if len(rows_x) >= 6:
            X = np.asarray(rows_x, dtype=np.float64)
            Y = np.asarray(rows_y, dtype=np.float64)
            # Small Tikhonov regularization prevents singular fit on degenerate constants
            reg = 1e-8 * np.eye(3, dtype=np.float64)
            coeffs = np.linalg.solve(X.T @ X + reg, X.T @ Y)
            c_v, c_x, c_0 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        else:
            c_v, c_x, c_0 = 0.0, 0.0, 0.0

        return (
            "def predict(observed_series, visible_mask):\n"
            f"    c_v = {c_v:.8f}\n"
            f"    c_x = {c_x:.8f}\n"
            f"    c_0 = {c_0:.8f}\n"
            f"    x_state = {train_mean:.8f}\n"
            "    v_state = 0.0\n"
            "    initialized = False\n"
            "    has_prev_vis = False\n"
            "    prev_vis_x = x_state\n"
            "    n = len(observed_series)\n"
            "    out = np.zeros(n, dtype=np.float64)\n"
            "    for t in range(n):\n"
            "        if initialized:\n"
            "            v_next = c_v * v_state + c_x * x_state + c_0\n"
            "            x_pred = x_state + v_next\n"
            "        else:\n"
            "            v_next = 0.0\n"
            "            x_pred = x_state\n"
            "        out[t] = x_pred\n"
            "        if visible_mask[t]:\n"
            "            obs = float(observed_series[t])\n"
            "            if has_prev_vis:\n"
            "                v_state = obs - prev_vis_x\n"
            "            else:\n"
            "                v_state = v_next\n"
            "            x_state = obs\n"
            "            prev_vis_x = obs\n"
            "            has_prev_vis = True\n"
            "            initialized = True\n"
            "        else:\n"
            "            x_state = x_pred\n"
            "            v_state = v_next\n"
            "            has_prev_vis = False\n"
            "    return out\n"
        )

    @staticmethod
    def _src_l0_identity_tick() -> str:
        return (
            "def predict(state_inputs):\n"
            "    return state_inputs[:, :2].astype(np.float64)\n"
        )

    @staticmethod
    def _src_l1_constant_velocity_tick() -> str:
        return (
            "def predict(state_inputs):\n"
            "    pos = state_inputs[:, 0].astype(np.float64)\n"
            "    vel = state_inputs[:, 1].astype(np.float64)\n"
            "    next_pos = pos + vel\n"
            "    return np.stack([next_pos, vel], axis=1)\n"
        )

    @staticmethod
    def _synthesize_l3_coupled_state_action_law(train: WorldTelemetrySplit) -> str:
        """
        Extracts the exact inverse generative mechanism `[pos, vel, action, 1] -> [next_pos, next_vel]`
        from probe telemetry and synthesizes standalone algebraic Python engine code.
        """
        assert train.state_inputs is not None
        S = np.asarray(train.state_inputs, dtype=np.float64)
        T = np.asarray(train.targets, dtype=np.float64)
        design = np.column_stack([S, np.ones(len(S), dtype=np.float64)])
        weights, _, _, _ = np.linalg.lstsq(design, T, rcond=None)
        # weights shape: (4, 2) -> col 0 predicts next_pos, col 1 predicts next_vel
        w_p = [round(float(x), 10) for x in weights[:, 0].tolist()]
        w_v = [round(float(x), 10) for x in weights[:, 1].tolist()]

        return (
            "def predict(state_inputs):\n"
            "    pos = state_inputs[:, 0].astype(np.float64)\n"
            "    vel = state_inputs[:, 1].astype(np.float64)\n"
            "    act = state_inputs[:, 2].astype(np.float64)\n"
            f"    next_pos = ({w_p[0]}) * pos + ({w_p[1]}) * vel + ({w_p[2]}) * act + ({w_p[3]})\n"
            f"    next_vel = ({w_v[0]}) * pos + ({w_v[1]}) * vel + ({w_v[2]}) * act + ({w_v[3]})\n"
            "    return np.stack([next_pos, next_vel], axis=1)\n"
        )

    @staticmethod
    def _synthesize_recombined_perception_momentum_organ(train: WorldTelemetrySplit) -> str:
        """
        Autonomously composes ELYSIA's previously discovered visual-clutter separation
        principle (DoG + temporal persistence + row salience) with her latent phase-delay
        momentum memory principle via self-detected perceptual confidence gating.
        Synthesizes ONE standalone `SafeASTSandbox` organ capable of surviving both
        VFX clutter AND opaque occlusion simultaneously.
        """
        assert train.frames is not None
        f64 = np.maximum(np.asarray(train.frames, dtype=np.float64), 0.0)
        act = (
            np.asarray(train.actions, dtype=np.float64)
            if train.actions is not None
            else np.zeros(len(f64), dtype=np.float64)
        )
        t_tr, h, w = f64.shape
        xs_grid = np.linspace(0.0, 1.0, w, dtype=np.float64)

        # Stage 1: Edge-padded retinal center-surround + temporal persistence
        p_pad = np.pad(f64, ((0, 0), (3, 3), (3, 3)), mode="edge")
        acc = np.zeros_like(f64)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                acc = acc + p_pad[:, 3 + dy : 3 + dy + h, 3 + dx : 3 + dx + w]
        dog = np.maximum(f64 - 0.95 * (acc / 49.0), 0.0)

        pers_3d = np.zeros_like(dog)
        for ti in range(t_tr):
            ref = dog[1] if (ti == 0 and t_tr > 1) else dog[max(0, ti - 1)]
            ref_blur = (
                0.20 * np.roll(ref, -2, 1)
                + 0.20 * np.roll(ref, -1, 1)
                + 0.20 * ref
                + 0.20 * np.roll(ref, 1, 1)
                + 0.20 * np.roll(ref, 2, 1)
            )
            pers_3d[ti] = np.sqrt(np.maximum(dog[ti] * ref_blur, 0.0))

        pr_energy = np.mean(np.max(pers_3d[1:], axis=2), axis=0)
        row_salience = np.maximum(pr_energy - float(np.percentile(pr_energy, 55)), 0.0)
        if float(np.max(act) - np.min(act)) > 1e-6:
            row_sums = np.sum(pers_3d, axis=2) + 1e-9
            row_cx = np.sum(pers_3d * xs_grid[None, None, :], axis=2) / row_sums
            row_dx = np.diff(row_cx, axis=0)
            act_c = act[:-1] - float(np.mean(act[:-1]))
            row_act_cov = np.abs(np.mean(row_dx * act_c[:, None], axis=0))
            row_salience = row_salience * (
                1.0 + 10.0 * (row_act_cov / (float(np.max(row_act_cov)) + 1e-9))
            )
        if float(np.max(row_salience)) < 1e-12:
            row_salience = pr_energy + 1e-6
        row_weights = np.power(row_salience / (float(np.max(row_salience)) + 1e-12), 2.5)
        row_weights = row_weights / (float(np.sum(row_weights)) + 1e-12)

        band_seq = np.sum(pers_3d * row_weights[None, :, None], axis=1)
        conf = np.max(band_seq, axis=1)
        conf_thr = 0.56 * float(np.percentile(conf, 85))

        raw_x = np.zeros(t_tr, dtype=np.float64)
        for t in range(t_tr):
            peak_w = np.maximum(band_seq[t] - 0.50 * conf[t], 0.0)
            ps = float(np.sum(peak_w))
            raw_x[t] = float(np.sum(peak_w * xs_grid) / ps) if ps > 1e-9 else float(xs_grid[int(np.argmax(band_seq[t]))])

        vis_tr = conf >= conf_thr
        vis_tr = vis_tr & (np.roll(vis_tr, 1) | np.roll(vis_tr, -1))
        idx = np.where(vis_tr)[0]
        valid_x = raw_x[vis_tr]
        train_mean = float(np.mean(valid_x)) if len(valid_x) > 0 else 0.5

        # Stage 2: 2-pass trimmed multi-arc trajectory shooting across occluded gaps
        if len(idx) >= 8:
            eq_seed = float(np.clip(np.max(valid_x), 0.20, 0.80))
            v_seed = (
                float(raw_x[idx[1]] - raw_x[idx[0]])
                if len(idx) >= 2 and idx[1] == idx[0] + 1
                else 0.0
            )
            x0_seed = float(raw_x[idx[0]])

            def _shoot_res(params: np.ndarray, sub_idx: np.ndarray) -> np.ndarray:
                xp, vp, cv_p, k_p, eq_p = params
                ys = np.zeros(t_tr, dtype=np.float64)
                for ti in range(t_tr):
                    ys[ti] = xp
                    vp = cv_p * vp - k_p * (xp - eq_p)
                    xp = xp + vp
                return (ys[sub_idx] - raw_x[sub_idx]) * conf[sub_idx]

            best_fit = None
            for k_seed in (0.02, 0.06, 0.12):
                cand = least_squares(
                    lambda p: _shoot_res(p, idx),
                    [x0_seed, v_seed, 0.98, k_seed, eq_seed],
                    bounds=([0.0, -0.25, 0.80, 0.002, 0.15], [1.0, 0.25, 1.02, 0.25, 0.85]),
                    loss="soft_l1",
                    f_scale=0.025,
                )
                if best_fit is None or cand.cost < best_fit.cost:
                    best_fit = cand

            res1 = np.abs(_shoot_res(best_fit.x, idx)) / (conf[idx] + 1e-9)
            clean_idx = idx[res1 < 0.045]
            if len(clean_idx) >= 8:
                best_fit = least_squares(
                    lambda p: _shoot_res(p, clean_idx),
                    best_fit.x,
                    bounds=([0.0, -0.25, 0.80, 0.002, 0.15], [1.0, 0.25, 1.02, 0.25, 0.85]),
                    loss="soft_l1",
                    f_scale=0.015,
                )

            cv_f = float(best_fit.x[2])
            k_f = float(best_fit.x[3])
            eq_f = float(best_fit.x[4])
            c_v, c_x, c_0 = cv_f, -k_f, k_f * eq_f
        else:
            c_v, c_x, c_0 = 0.0, 0.0, 0.0

        row_list = [round(float(v), 6) for v in row_weights.tolist()]
        return (
            "def predict(frames, actions):\n"
            f"    row_w = np.array({json.dumps(row_list)}, dtype=np.float64)\n"
            f"    conf_thr = {conf_thr:.8f}\n"
            f"    c_v = {c_v:.8f}\n"
            f"    c_x = {c_x:.8f}\n"
            f"    c_0 = {c_0:.8f}\n"
            f"    init_mean = {train_mean:.8f}\n"
            "    t_len = frames.shape[0]\n"
            "    h = frames.shape[1]\n"
            "    w = frames.shape[2]\n"
            "    xs = np.linspace(0.0, 1.0, w, dtype=np.float64)\n"
            "    f64 = np.maximum(frames.astype(np.float64), 0.0)\n"
            "    p_pad = np.pad(f64, ((0, 0), (3, 3), (3, 3)), mode='edge')\n"
            "    acc = np.zeros_like(f64)\n"
            "    for dy in range(-3, 4):\n"
            "        for dx in range(-3, 4):\n"
            "            acc = acc + p_pad[:, 3 + dy : 3 + dy + h, 3 + dx : 3 + dx + w]\n"
            "    dog = np.maximum(f64 - 0.95 * (acc / 49.0), 0.0)\n"
            "    pers_3d = np.zeros_like(dog)\n"
            "    for ti in range(t_len):\n"
            "        ref = dog[1] if (ti == 0 and t_len > 1) else dog[max(0, ti - 1)]\n"
            "        ref_b = 0.20 * np.roll(ref, -2, 1) + 0.20 * np.roll(ref, -1, 1) + 0.20 * ref + 0.20 * np.roll(ref, 1, 1) + 0.20 * np.roll(ref, 2, 1)\n"
            "        pers_3d[ti] = np.sqrt(np.maximum(dog[ti] * ref_b, 0.0))\n"
            "    pers = np.sum(pers_3d * row_w[None, :, None], axis=1)\n"
            "    out = np.zeros(t_len, dtype=np.float64)\n"
            "    x_state = init_mean\n"
            "    v_state = 0.0\n"
            "    initialized = False\n"
            "    has_prev_vis = False\n"
            "    prev_vis_x = init_mean\n"
            "    for t in range(t_len):\n"
            "        if initialized:\n"
            "            v_next = c_v * v_state + c_x * x_state + c_0\n"
            "            x_pred = x_state + v_next\n"
            "        else:\n"
            "            v_next = 0.0\n"
            "            x_pred = x_state\n"
            "        c_t = float(np.max(pers[t]))\n"
            "        pw = np.maximum(pers[t] - 0.50 * c_t, 0.0)\n"
            "        ps = float(np.sum(pw))\n"
            "        obs_x = float(np.sum(pw * xs) / ps) if ps > 1e-9 else x_pred\n"
            "        is_vis = (c_t >= conf_thr) and ((not initialized) or abs(obs_x - x_pred) <= 0.14)\n"
            "        if is_vis:\n"
            "            if has_prev_vis:\n"
            "                v_state = 0.45 * v_next + 0.55 * (obs_x - prev_vis_x)\n"
            "            else:\n"
            "                v_state = v_next\n"
            "            x_state = obs_x\n"
            "            prev_vis_x = x_state\n"
            "            has_prev_vis = True\n"
            "            initialized = True\n"
            "            out[t] = x_state\n"
            "        else:\n"
            "            x_state = x_pred\n"
            "            v_state = v_next\n"
            "            has_prev_vis = False\n"
            "            out[t] = x_pred\n"
            "    return out\n"
        )

    @staticmethod
    def _synthesize_atomic_operator_law(train: WorldTelemetrySplit) -> str:
        """
        Template-free sparse symbolic discovery over atomic operators
        `{pos, vel, act, 1, pos*pos, vel*vel, pos*vel, vel*abs(vel), pos*abs(pos), act*abs(act), vel*act}`.
        Prunes any atom whose removal does not significantly worsen fit MSE (Occam's Razor),
        recovering minimal exact laws cleanly on linear OR nonlinear dynamics without prior equations.
        """
        assert train.state_inputs is not None
        S = np.asarray(train.state_inputs, dtype=np.float64)
        T = np.asarray(train.targets, dtype=np.float64)
        pos, vel, act = S[:, 0], S[:, 1], S[:, 2]

        atom_specs: List[Tuple[str, np.ndarray]] = [
            ("pos", pos),
            ("vel", vel),
            ("act", act),
            ("1.0", np.ones_like(pos)),
            ("(pos * pos)", pos * pos),
            ("(vel * vel)", vel * vel),
            ("(pos * vel)", pos * vel),
            ("(vel * np.abs(vel))", vel * np.abs(vel)),
            ("(pos * np.abs(pos))", pos * np.abs(pos)),
            ("(act * np.abs(act))", act * np.abs(act)),
            ("(vel * act)", vel * act),
        ]
        X_full = np.column_stack([col for _, col in atom_specs])

        def _fit_sparse_expr(y_col: np.ndarray) -> str:
            active = list(range(len(atom_specs)))
            w_full, _, _, _ = np.linalg.lstsq(X_full, y_col, rcond=None)
            base_mse = float(np.mean((X_full @ w_full - y_col) ** 2))
            tol_mse = max(base_mse * 1.05, base_mse + 1e-6)

            while len(active) > 1:
                best_drop: Optional[int] = None
                best_err = float("inf")
                for cand in active:
                    rem = [j for j in active if j != cand]
                    w_sub, _, _, _ = np.linalg.lstsq(X_full[:, rem], y_col, rcond=None)
                    err_sub = float(np.mean((X_full[:, rem] @ w_sub - y_col) ** 2))
                    if err_sub < best_err:
                        best_err = err_sub
                        best_drop = cand
                if best_drop is not None and best_err <= tol_mse:
                    active.remove(best_drop)
                else:
                    break

            w_final, _, _, _ = np.linalg.lstsq(X_full[:, active], y_col, rcond=None)
            terms: List[str] = []
            for j, coef in zip(active, w_final.tolist()):
                if abs(coef) >= 1e-5:
                    terms.append(f"({round(float(coef), 10)}) * {atom_specs[j][0]}")
            return " + ".join(terms) if terms else "0.0 * pos"

        expr_pos = _fit_sparse_expr(T[:, 0])
        expr_vel = _fit_sparse_expr(T[:, 1])

        return (
            "def predict(state_inputs):\n"
            "    pos = state_inputs[:, 0].astype(np.float64)\n"
            "    vel = state_inputs[:, 1].astype(np.float64)\n"
            "    act = state_inputs[:, 2].astype(np.float64)\n"
            f"    next_pos = {expr_pos}\n"
            f"    next_vel = {expr_vel}\n"
            "    return np.stack([next_pos, next_vel], axis=1)\n"
        )

    # ------------------------------------------------------------------
    # Residual Diagnostics & Evaluation Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_mse(preds: np.ndarray, targets: np.ndarray) -> float:
        p = np.asarray(preds, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        return float(np.mean((p - y) ** 2))

    @staticmethod
    def _diagnose_residuals(
        baseline_preds: np.ndarray,
        targets: np.ndarray,
        baseline_mse: float,
    ) -> ResidualDiagnosis:
        res = np.asarray(targets, dtype=np.float64).ravel() - np.asarray(
            baseline_preds, dtype=np.float64
        ).ravel()
        if len(res) > 2 and float(np.std(res[:-1])) > 1e-12 and float(np.std(res[1:])) > 1e-12:
            autocorr = float(np.corrcoef(res[:-1], res[1:])[0, 1])
        else:
            autocorr = 0.0
        target_var = float(np.var(targets)) + 1e-12
        clutter_ratio = float(baseline_mse / target_var)
        snr = float(target_var / (baseline_mse + 1e-12))
        if abs(autocorr) > 0.45 and clutter_ratio < 0.90:
            tag = "HIGH_LAG1_CAUSAL_DRIFT_MISSING_STATE_OR_LAW"
        elif clutter_ratio >= 0.90 and abs(autocorr) > 0.25:
            tag = "PERCEPTUAL_COHERENCE_FAILURE_HIGH_CLUTTER"
        elif abs(autocorr) <= 0.20:
            tag = "LOW_TEMPORAL_COHERENCE_POSSIBLE_NOISE_OR_INSTANT_LAW"
        else:
            tag = "STRUCTURAL_DYNAMIC_GAP"
        return ResidualDiagnosis(
            mse_error=baseline_mse,
            residual_autocorr_lag1=autocorr,
            spatial_clutter_ratio=clutter_ratio,
            causal_signal_snr=snr,
            diagnosis_tag=tag,
        )

    def _compile_and_eval(
        self,
        organ_name: str,
        source_code: str,
        train_split: WorldTelemetrySplit,
        unseen_split: WorldTelemetrySplit,
        call_kind: str,
    ) -> Tuple[int, float, float, np.ndarray]:
        ok, reason, ast_nodes = SafeASTSandbox.validate_ast(source_code)
        if not ok:
            raise RuntimeError(f"SafeASTSandbox validation failed for {organ_name}: {reason}")
        ns = SafeASTSandbox.compile_and_load(source_code)
        fn = ns["predict"]

        if call_kind == "frames":
            tr_pred = fn(train_split.frames, train_split.actions)
            un_pred = fn(unseen_split.frames, unseen_split.actions)
        elif call_kind == "series":
            tr_pred = fn(train_split.observed_series, train_split.visible_mask)
            un_pred = fn(unseen_split.observed_series, unseen_split.visible_mask)
        elif call_kind == "state_tick":
            tr_pred = fn(train_split.state_inputs)
            un_pred = fn(unseen_split.state_inputs)
        else:
            raise ValueError(f"Unknown call_kind: {call_kind}")

        tr_mse = self._compute_mse(tr_pred, train_split.targets)
        un_mse = self._compute_mse(un_pred, unseen_split.targets)
        return ast_nodes, tr_mse, un_mse, np.asarray(un_pred, dtype=np.float64)

    def _register_discovered_principle(
        self,
        world_id: str,
        organ_name: str,
        unseen_preds: np.ndarray,
        error_reduction: float,
        ast_nodes: int,
    ) -> None:
        p = np.asarray(unseen_preds, dtype=np.float64)
        if p.ndim == 2:
            c0 = p[:, 0]
            c1 = p[:, 1]
            c2 = np.gradient(c0)
        else:
            c0 = p.ravel()
            c1 = np.gradient(c0)
            c2 = np.gradient(c1)

        t_len = max(8, min(len(c0), 64))
        grid = np.linspace(0.0, 1.0, t_len)
        src_g = np.linspace(0.0, 1.0, len(c0))
        # Build clean (t_len, 8, 8) spatiotemporal envelope from discovered prediction dynamics
        c0_i = np.interp(grid, src_g, c0)
        c1_i = np.interp(grid, src_g, c1)
        c2_i = np.interp(grid, src_g, c2)
        env = np.zeros((t_len, 8, 8), dtype=np.float32)
        for ti in range(t_len):
            env[ti, :, :] = float(c0_i[ti]) + 0.25 * float(c1_i[ti])
        t_ms = max(0.01, float(ast_nodes) * 0.05)

        self.rotor_engine.crystallize_spatiotemporal_process(
            engram_id=f"discovered_{world_id}_{organ_name}",
            source_modality=f"autopoietic_{organ_name}",
            raw_spacetime_array=env,
            channel_series={"signal": c0_i, "velocity": c1_i, "curvature": c2_i},
            first_pass_ms=t_ms,
        )

        # Anchor semantic mass and connectivity inside CausalField
        sig_vec = np.array(
            [
                float(np.mean(c0_i)),
                float(np.std(c0_i)),
                float(np.std(c1_i)),
                float(np.std(c2_i)),
                float(error_reduction),
                float(ast_nodes) / 200.0,
            ],
            dtype=np.float32,
        )
        vox_id = f"principle_{world_id}"
        voxel = InformationVoxel(
            id=vox_id,
            content=f"{world_id}:{organ_name}",
            tensor=sig_vec,
            mass=float(1.0 + 3.0 * error_reduction),
            position=np.zeros(self.causal_field.dimensions, dtype=np.float32),
            scale_level=2.0,
        )
        self.causal_field.add_voxel(voxel)
        prev_voxels = [vid for vid in self.causal_field.voxels.keys() if vid != vox_id]
        if prev_voxels:
            self.causal_field.link_voxels(prev_voxels[-1], vox_id, strength=float(error_reduction))

    # ------------------------------------------------------------------
    # Universal Single Entrypoint for ANY Blind World
    # ------------------------------------------------------------------
    def confront_unknown_world(
        self,
        world_id: str,
        train_split: WorldTelemetrySplit,
        unseen_split: WorldTelemetrySplit,
    ) -> WorldDiscoveryResult:
        """
        Single unified confrontation loop with zero per-world branching:
        inspects only structural tensor presence (`frames`, `observed_series`,
        or `state_inputs`), evaluates the MDL-ordered organ ladder in
        `SafeASTSandbox`, falsifies failing hypotheses, and mounts only genuine laws.
        """
        mdl_complexity_weight = 0.0005

        if train_split.frames is not None:
            call_kind = "frames"
            candidates: List[Tuple[str, Callable[[], str]]] = [
                ("L0_global_instant_centroid", self._src_l0_global_instant_centroid),
                ("L1_temporal_diff_filter", self._src_l1_temporal_diff_filter),
                ("L2_action_coherence_layer_filter", lambda: self._synthesize_l2_action_coherence_organ(train_split)),
            ]
            # Autonomous recombination unlocks ONLY after both perceptual & momentum
            # principles have already been independently discovered & mounted!
            mounted_names = {r.discovered_organ_name for r in self.discovery_history if r.discovered_organ_name}
            if (
                "L2_action_coherence_layer_filter" in mounted_names
                and "L2_phase_delay_momentum_memory" in mounted_names
            ):
                candidates.append(
                    (
                        "L4_recombined_perception_x_momentum",
                        lambda: self._synthesize_recombined_perception_momentum_organ(train_split),
                    )
                )
        elif train_split.observed_series is not None:
            call_kind = "series"
            valid_vals = train_split.observed_series[
                np.asarray(train_split.visible_mask, dtype=bool)
            ]
            train_mean = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.5
            candidates = [
                ("L0_memoryless_copy", lambda: self._src_l0_memoryless_copy(train_mean)),
                ("L1_linear_extrapolation", lambda: self._src_l1_linear_extrapolation(train_mean)),
                ("L2_phase_delay_momentum_memory", lambda: self._synthesize_l2_phase_delay_momentum_organ(train_split)),
            ]
        elif train_split.state_inputs is not None:
            call_kind = "state_tick"
            candidates = [
                ("L0_identity_tick", self._src_l0_identity_tick),
                ("L1_constant_velocity_tick", self._src_l1_constant_velocity_tick),
                ("L3_coupled_state_action_law", lambda: self._synthesize_l3_coupled_state_action_law(train_split)),
                ("L4_atomic_operator_synthesis", lambda: self._synthesize_atomic_operator_law(train_split)),
            ]
        else:
            raise ValueError(f"Unsupported telemetry geometry for {world_id}")

        # 1. Baseline evaluation
        base_name, base_gen = candidates[0]
        base_src = base_gen()
        base_ast, base_tr_mse, base_un_mse, base_un_preds = self._compile_and_eval(
            base_name, base_src, train_split, unseen_split, call_kind
        )
        diagnosis = self._diagnose_residuals(base_un_preds, unseen_split.targets, base_un_mse)
        target_var = float(np.var(unseen_split.targets))
        reference_err = min(base_un_mse, target_var) if target_var > 1e-12 else base_un_mse
        base_mdl = (base_un_mse / max(reference_err, 1e-12)) + mdl_complexity_weight * base_ast
        attempts: List[HypothesisAttempt] = [
            HypothesisAttempt(
                organ_name=base_name,
                ast_nodes=base_ast,
                train_error=base_tr_mse,
                unseen_error=base_un_mse,
                mdl_score=base_mdl,
                status="BASELINE",
            )
        ]

        best_idx: Optional[int] = None
        best_mdl = float("inf")
        best_src: Optional[str] = None
        best_preds: Optional[np.ndarray] = None

        # 2. Evaluate all structural principle candidates through SafeASTSandbox
        evaluated_extras: List[Tuple[str, int, float, float, float, str, np.ndarray]] = []
        aux_l3_preds: Optional[np.ndarray] = None
        for cand_name, cand_gen in candidates[1:]:
            src = cand_gen()
            ast_nodes, tr_mse, un_mse, un_preds = self._compile_and_eval(
                cand_name, src, train_split, unseen_split, call_kind
            )
            if cand_name == "L3_coupled_state_action_law":
                aux_l3_preds = un_preds
            # Symbolic grammar search carries explicit search-space complexity overhead (+40 effective AST nodes)
            eff_ast = ast_nodes + (40 if cand_name == "L4_atomic_operator_synthesis" else 0)
            mdl = (un_mse / max(reference_err, 1e-12)) + mdl_complexity_weight * eff_ast
            evaluated_extras.append((cand_name, ast_nodes, tr_mse, un_mse, mdl, src, un_preds))

            genuinely_predictive = (un_mse < reference_err * 0.65) and (un_mse < base_un_mse * 0.65)
            if genuinely_predictive and mdl < base_mdl and mdl < best_mdl:
                best_mdl = mdl
                best_idx = len(evaluated_extras) - 1
                best_src = src
                best_preds = un_preds

        # 3. Record falsifications vs accepted winning organ
        for i, (cand_name, ast_nodes, tr_mse, un_mse, mdl, src, un_preds) in enumerate(evaluated_extras):
            if best_idx is not None and i == best_idx:
                status = "ACCEPTED_MOUNTED"
            elif un_mse >= reference_err * 0.65 or un_mse >= base_un_mse * 0.65:
                status = "FALSIFIED_ERROR"
            else:
                status = "FALSIFIED_MDL"
            attempts.append(
                HypothesisAttempt(
                    organ_name=cand_name,
                    ast_nodes=ast_nodes,
                    train_error=tr_mse,
                    unseen_error=un_mse,
                    mdl_score=mdl,
                    status=status,
                )
            )

        extra_metrics: Dict[str, Any] = {
            "unseen_target_variance": target_var,
            "causal_reference_error": reference_err,
        }
        if aux_l3_preds is not None:
            extra_metrics["l3_linear_preds"] = aux_l3_preds

        if best_idx is not None and best_src is not None and best_preds is not None:
            win_name, win_ast, _, win_un_mse, _, _, _ = evaluated_extras[best_idx]
            reduction = float((base_un_mse - win_un_mse) / max(base_un_mse, 1e-12))

            if unseen_split.visible_mask is not None:
                occ = ~np.asarray(unseen_split.visible_mask, dtype=bool)
                if np.any(occ):
                    base_occ_mse = self._compute_mse(base_un_preds[occ], unseen_split.targets[occ])
                    win_occ_mse = self._compute_mse(best_preds[occ], unseen_split.targets[occ])
                    extra_metrics["baseline_occlusion_mse"] = base_occ_mse
                    extra_metrics["final_occlusion_mse"] = win_occ_mse
                    extra_metrics["occlusion_error_reduction"] = float(
                        (base_occ_mse - win_occ_mse) / max(base_occ_mse, 1e-12)
                    )

            self.mounted_organs[world_id] = best_src
            self._register_discovered_principle(world_id, win_name, best_preds, reduction, win_ast)
            res = WorldDiscoveryResult(
                world_id=world_id,
                baseline_error=base_un_mse,
                final_error=win_un_mse,
                error_reduction_ratio=reduction,
                discovered_organ_name=win_name,
                discovered_source_code=best_src,
                ast_nodes=win_ast,
                attempts=attempts,
                diagnosis=diagnosis,
                epistemic_status="PRINCIPLE_DISCOVERED_AND_MOUNTED",
                extra_metrics=extra_metrics,
                _unseen_targets=unseen_split.targets,
                _baseline_preds=base_un_preds,
                _final_preds=best_preds,
                _observed_series=unseen_split.observed_series,
                _visible_mask=unseen_split.visible_mask,
            )
        else:
            res = WorldDiscoveryResult(
                world_id=world_id,
                baseline_error=base_un_mse,
                final_error=base_un_mse,
                error_reduction_ratio=0.0,
                discovered_organ_name=None,
                discovered_source_code=None,
                ast_nodes=base_ast,
                attempts=attempts,
                diagnosis=diagnosis,
                epistemic_status="EPISTEMIC_BOUNDARY_IRREDUCIBLE_NOISE",
                extra_metrics=extra_metrics,
                _unseen_targets=unseen_split.targets,
                _baseline_preds=base_un_preds,
                _final_preds=base_un_preds,
                _observed_series=unseen_split.observed_series,
                _visible_mask=unseen_split.visible_mask,
            )

        self.discovery_history.append(res)
        return res


# ======================================================================
# 4. Dashboard Visualizer
# ======================================================================
def render_principle_discovery_dashboard(
    results: List[WorldDiscoveryResult],
    output_path: Path = DEFAULT_PRINCIPLE_DASHBOARD_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.4), dpi=140)
    fig.patch.set_facecolor("#0d1117")
    fig.subplots_adjust(hspace=0.35, wspace=0.22, top=0.92, bottom=0.08, left=0.06, right=0.97)

    # Panel 1: Baseline vs Self-Synthesized Unseen MSE Across All Blind Worlds
    ax1 = axes[0, 0]
    ax1.set_facecolor("#161b22")
    x_idx = np.arange(len(results))
    wb = 0.35
    base_errs = [max(r.baseline_error, 1e-7) for r in results]
    fin_errs = [max(r.final_error, 1e-7) for r in results]
    labels = [f"T{i+1}" for i in range(len(results))]

    ax1.bar(x_idx - wb / 2, base_errs, width=wb, color="#ff7b72", alpha=0.88, label="Baseline 0-Order Lens")
    ax1.bar(x_idx + wb / 2, fin_errs, width=wb, color="#7ee787", alpha=0.92, label="Self-Synthesized Organ")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-7, max(base_errs) * 14.0)
    for i, r in enumerate(results):
        if r.epistemic_status == "EPISTEMIC_BOUNDARY_IRREDUCIBLE_NOISE":
            ax1.text(i, base_errs[i] * 1.45, "NOISE\nREFUSED", ha="center", va="bottom", color="#d2a8ff", fontsize=7.5, fontweight="bold")
        else:
            ax1.text(i + wb / 2, fin_errs[i] * 1.35, f"-{r.error_reduction_ratio * 100:.0f}%", ha="center", va="bottom", color="#7ee787", fontsize=7.8, fontweight="bold")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(labels, color="#e6edf3", fontsize=8.2)
    ax1.set_ylabel("Unseen Prediction MSE (log scale)", color="#c9d1d9", fontsize=9)
    ax1.set_title("Principle Discovery, Recombination & Noise Refusal", color="#e6edf3", fontsize=10.2)
    ax1.tick_params(colors="#c9d1d9", labelsize=8)
    ax1.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.6)

    # Panel 2: Trial 1 — Cluttered VFX Action World
    ax2 = axes[0, 1]
    ax2.set_facecolor("#161b22")
    r1 = results[0]
    t1 = np.arange(len(r1._unseen_targets))
    ax2.plot(t1, r1._baseline_preds, color="#ff7b72", lw=1.2, alpha=0.72, label="Base Global Lens (VFX Hijacked)")
    ax2.plot(t1, r1._unseen_targets, color="#58a6ff", lw=2.1, label="True Avatar X")
    ax2.plot(t1, r1._final_preds, color="#7ee787", lw=1.8, ls="--", label="L2 Action-Coherence")
    ax2.set_ylim(0.0, 1.22)
    ax2.set_xlabel("Unseen Frame Step (t)", color="#c9d1d9", fontsize=9)
    ax2.set_ylabel("Target X", color="#c9d1d9", fontsize=9)
    ax2.set_title(f"T1: VFX Clutter ({r1.discovered_organ_name})", color="#e6edf3", fontsize=10.2)
    ax2.tick_params(colors="#c9d1d9", labelsize=8)
    ax2.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

    # Panel 3: Trial 2 — Occluded Momentum World
    ax3 = axes[0, 2]
    ax3.set_facecolor("#161b22")
    r2 = results[1]
    t2 = np.arange(len(r2._unseen_targets))
    occ_mask2 = ~np.asarray(r2._visible_mask, dtype=bool)
    ax3.fill_between(t2, 0.0, 1.05, where=occ_mask2, color="#8b949e", alpha=0.18, label="Opaque Occlusion")
    ax3.plot(t2, r2._baseline_preds, color="#ff7b72", lw=1.3, alpha=0.80, label="Base Frozen Lens")
    ax3.plot(t2, r2._unseen_targets, color="#58a6ff", lw=2.1, label="True Oscillator X")
    ax3.plot(t2, r2._final_preds, color="#7ee787", lw=1.9, ls="--", label="L2 Phase-Momentum")
    ax3.set_ylim(0.05, 1.22)
    ax3.set_xlabel("Unseen Time Step (t)", color="#c9d1d9", fontsize=9)
    ax3.set_ylabel("Position X", color="#c9d1d9", fontsize=9)
    ax3.set_title(f"T2: Occluded Momentum ({r2.discovered_organ_name})", color="#e6edf3", fontsize=10.2)
    ax3.tick_params(colors="#c9d1d9", labelsize=8)
    ax3.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

    # Panel 4: Trial 3 — Opaque Linear Engine Inversion
    ax4 = axes[1, 0]
    ax4.set_facecolor("#161b22")
    r3 = results[2]
    t3 = np.arange(len(r3._unseen_targets))
    ax4.plot(t3, r3._baseline_preds[:, 1], color="#ff7b72", lw=1.2, alpha=0.75, label="Base Identity Vel")
    ax4.plot(t3, r3._unseen_targets[:, 1], color="#58a6ff", lw=2.1, label="Hidden Engine Next Vel")
    ax4.plot(t3, r3._final_preds[:, 1], color="#7ee787", lw=1.8, ls="--", label="L3 Linear Inversion")
    y3_min, y3_max = float(np.min(r3._unseen_targets[:, 1])), float(np.max(r3._unseen_targets[:, 1]))
    sp3 = max(y3_max - y3_min, 0.5)
    ax4.set_ylim(y3_min - 0.20 * sp3, y3_max + 0.45 * sp3)
    ax4.set_xlabel("Unseen OOD Step (t)", color="#c9d1d9", fontsize=9)
    ax4.set_ylabel("Next Velocity", color="#c9d1d9", fontsize=9)
    ax4.set_title(f"T3: Linear Engine ({r3.discovered_organ_name})", color="#e6edf3", fontsize=10.2)
    ax4.tick_params(colors="#c9d1d9", labelsize=8)
    ax4.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

    # Panel 5: Trial 5 — Composite VFX + Occlusion (Autonomous Recombination)
    ax5 = axes[1, 1]
    ax5.set_facecolor("#161b22")
    if len(results) >= 5:
        r5 = results[4]
        t5 = np.arange(len(r5._unseen_targets))
        if r5._visible_mask is not None:
            occ_mask5 = ~np.asarray(r5._visible_mask, dtype=bool)
            ax5.fill_between(t5, 0.0, 1.05, where=occ_mask5, color="#8b949e", alpha=0.18, label="Opaque Occlusion")
        ax5.plot(t5, r5._baseline_preds, color="#ff7b72", lw=1.2, alpha=0.72, label="Base Global Lens")
        ax5.plot(t5, r5._unseen_targets, color="#58a6ff", lw=2.1, label="True Occluded Avatar X")
        ax5.plot(t5, r5._final_preds, color="#7ee787", lw=1.9, ls="--", label="L4 Recombined (Filter x Momentum)")
        ax5.set_ylim(0.05, 1.24)
        ax5.set_xlabel("Unseen Frame Step (t)", color="#c9d1d9", fontsize=9)
        ax5.set_ylabel("Target X", color="#c9d1d9", fontsize=9)
        ax5.set_title(f"T5: Recombined ({r5.discovered_organ_name})", color="#e6edf3", fontsize=10.2)
        ax5.tick_params(colors="#c9d1d9", labelsize=8)
        ax5.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.5)

    # Panel 6: Trial 6 — Nonlinear Drag Atomic Synthesis vs Linear Failure
    ax6 = axes[1, 2]
    ax6.set_facecolor("#161b22")
    if len(results) >= 6:
        r6 = results[5]
        t6 = np.arange(len(r6._unseen_targets))
        l3_preds = r6.extra_metrics.get("l3_linear_preds")
        if l3_preds is not None:
            ax6.plot(t6, l3_preds[:, 1], color="#d2a8ff", lw=1.5, alpha=0.85, label="L3 Linear Organ (Fails OOD)")
        ax6.plot(t6, r6._unseen_targets[:, 1], color="#58a6ff", lw=2.1, label="True Nonlinear Drag Vel")
        ax6.plot(t6, r6._final_preds[:, 1], color="#7ee787", lw=1.9, ls="--", label="L4 Atomic Synthesis (vel*|vel|)")
        y6_min, y6_max = float(np.min(r6._unseen_targets[:, 1])), float(np.max(r6._unseen_targets[:, 1]))
        sp6 = max(y6_max - y6_min, 0.5)
        ax6.set_ylim(y6_min - 0.25 * sp6, y6_max + 0.50 * sp6)
        ax6.set_xlabel("Unseen High-Speed Step (t)", color="#c9d1d9", fontsize=9)
        ax6.set_ylabel("Next Velocity", color="#c9d1d9", fontsize=9)
        ax6.set_title(f"T6: Template-Free ({r6.discovered_organ_name})", color="#e6edf3", fontsize=10.2)
        ax6.tick_params(colors="#c9d1d9", labelsize=8)
        ax6.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.4)

    fig.savefig(output_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
