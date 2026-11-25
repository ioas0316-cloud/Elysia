r"""
ELYSIA Pure Uncontrolled Self-Modifying Causal Organism
======================================================

A genuinely self-modifying causal system with ZERO hardcoded scenarios, ZERO domain
lists, and ZERO canned textual conclusions:

1. Senses continuous physical state $S_t \in \mathbb{R}^{10}$ from live reality
   (`ScreenFrameSnapshot`, OS window, real pixels/files) and projects it through
   a mutable perceptual matrix $P \in \mathbb{R}^{D \times 10}$.
2. Chooses its own action $a_t$ and predicts $\hat{S}_{t+1} = S_t + W_a [S_t, 1]$
   BEFORE acting.
3. Contrasts reality $S_{t+1}^{\text{real}}$ against $\hat{S}_{t+1}$ along 4 axes:
   **What** matched/diverged, **How much** ($L_2$), **How** (signed direction),
   and **Why** (`unseen_action_effect`, `external_world_autonomy`,
   `missing_sensory_axis`, `model_parameter_drift`, `aligned_with_reality`).
4. **Self-Modifies**: updates $W_a$, expands $P$ with orthogonal emergent axes on
   persistent residuals, retroactively re-evaluates past episodes through its new
   self, counterfactually simulates alternative actions to revise policy, and
   **automatically rolls back** any mutation that worsens real prediction error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.embodiment.pc_sensory_motor_body import (
    ScreenFrameSnapshot,
    ScreenRetinaSensor,
    WindowsForegroundSensor,
    inspect_local_file_sensory,
)


BASE_AXIS_NAMES: Tuple[str, ...] = (
    "luma_mean",
    "r_mean",
    "g_mean",
    "b_mean",
    "contrast_std",
    "warm_cool_norm",
    "motion_flux",
    "edge_sharpness",
    "salient_x",
    "salient_y",
)

AUTONOMOUS_ACTIONS: Tuple[str, ...] = (
    "observe_still",
    "zoom_salient_focus",
    "zoom_opposite_quadrant",
    "probe_workspace_file",
)


def encode_live_world_state(
    snap: ScreenFrameSnapshot,
    *,
    edge_override: Optional[float] = None,
    contrast_override: Optional[float] = None,
    rgb_override: Optional[Sequence[float]] = None,
    warm_cool_override: Optional[float] = None,
) -> np.ndarray:
    """Encodes real sensory frame measurements into a continuous 10-D physical state vector."""
    rgb = np.asarray(snap.frame_small, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        rgb = np.zeros((24, 32, 3), dtype=np.float64)

    if rgb_override is not None and len(rgb_override) == 3:
        r_m, g_m, b_m = (float(rgb_override[0]), float(rgb_override[1]), float(rgb_override[2]))
    else:
        r_m = float(np.mean(rgb[..., 0]))
        g_m = float(np.mean(rgb[..., 1]))
        b_m = float(np.mean(rgb[..., 2]))

    luma_m = float(np.clip(0.2126 * r_m + 0.7152 * g_m + 0.0722 * b_m, 0.0, 1.0))
    contrast_v = float(contrast_override if contrast_override is not None else snap.contrast_std)
    warm_raw = float(warm_cool_override if warm_cool_override is not None else snap.warm_cool_ratio)
    warm_norm = float(np.clip(warm_raw / 2.0, 0.0, 1.0))
    motion_v = float(np.clip(snap.motion_flux, 0.0, 1.0))

    if edge_override is not None:
        edge_v = float(np.clip(edge_override, 0.0, 1.0))
    else:
        luma_img = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        dx = float(np.mean(np.abs(np.diff(luma_img, axis=1)))) if luma_img.shape[1] > 1 else 0.0
        dy = float(np.mean(np.abs(np.diff(luma_img, axis=0)))) if luma_img.shape[0] > 1 else 0.0
        edge_v = float(np.clip((dx + dy) * 3.0, 0.0, 1.0))

    sx = float(np.clip(snap.salient_xy_norm[0], 0.0, 1.0))
    sy = float(np.clip(snap.salient_xy_norm[1], 0.0, 1.0))

    return np.array(
        [luma_m, r_m, g_m, b_m, contrast_v, warm_norm, motion_v, edge_v, sx, sy],
        dtype=np.float64,
    )


@dataclass
class DiscrepancyContrast:
    """4-axis mismatch analysis between internal prediction and real world outcome."""

    what_matched: List[str]
    what_diverged: List[str]
    how_much_l2: float
    how_signed_deltas: Dict[str, float]
    why_attribution: str  # "unseen_action_effect" | "external_world_autonomy" | "missing_sensory_axis" | "model_parameter_drift" | "aligned_with_reality"
    why_detail: str


@dataclass
class SelfModificationRecord:
    """Actual structural/weight mutation applied after confronting external reality."""

    cycle: int
    action_taken: str
    mutation_kind: str  # "weight_update" | "axis_expansion" | "rollback" | "steady_alignment"
    parameter_delta_norm: float
    past_error_before: float
    past_error_after: float
    counterfactual_best_action: str
    new_axis_name: str = ""
    kept: bool = True


@dataclass
class OrganismCycleTrace:
    """Transparent raw telemetry of one uncontrolled autonomous perception-prediction-mutation cycle."""

    cycle: int
    action_chosen: str
    predicted_next: List[float]
    actual_next: List[float]
    contrast: DiscrepancyContrast
    modification: SelfModificationRecord
    reobserved_error_norm: Optional[float]
    active_axes_count: int
    window_title: str


class SelfModifyingCausalOrganism:
    r"""
    Pure uncontrolled self-modifying organism operating on real continuous vectors.
    Starts with zero knowledge of external action transitions ($\hat{S}_{t+1} = S_t$).
    """

    MAX_EPISODES = 200
    MAX_AXES = 20

    def __init__(self, actions: Sequence[str] = AUTONOMOUS_ACTIONS) -> None:
        self.actions: List[str] = list(actions)
        self.axis_names: List[str] = list(BASE_AXIS_NAMES)
        self.projection_matrix: np.ndarray = np.eye(len(BASE_AXIS_NAMES), dtype=np.float64)
        self.transition_weights: Dict[str, np.ndarray] = {
            act: np.zeros((len(self.axis_names), len(self.axis_names) + 1), dtype=np.float64)
            for act in self.actions
        }
        self.action_counts: Dict[str, int] = {act: 0 for act in self.actions}
        self.action_policy_values: Dict[str, float] = {act: 0.0 for act in self.actions}
        self.action_lr_scales: Dict[str, float] = {act: 1.0 for act in self.actions}
        self.residual_history: Dict[str, List[List[float]]] = {act: [] for act in self.actions}
        self.episode_buffer: List[Dict[str, Any]] = []
        self._pending_verification: Dict[str, Dict[str, Any]] = {}
        self._last_error_by_action: Dict[str, float] = {}
        self.mutations_kept: int = 0
        self.mutations_reverted: int = 0
        self.cycle_count: int = 0
        self._probe_file_idx: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actions": list(self.actions),
            "axis_names": list(self.axis_names),
            "projection_matrix": self.projection_matrix.tolist(),
            "transition_weights": {k: v.tolist() for k, v in self.transition_weights.items()},
            "action_counts": dict(self.action_counts),
            "action_policy_values": dict(self.action_policy_values),
            "action_lr_scales": dict(self.action_lr_scales),
            "residual_history": dict(self.residual_history),
            "episode_buffer": list(self.episode_buffer),
            "pending_verification": {
                k: {
                    **v,
                    "weights_before": np.asarray(v["weights_before"]).tolist(),
                }
                for k, v in self._pending_verification.items()
            },
            "last_error_by_action": dict(self._last_error_by_action),
            "mutations_kept": self.mutations_kept,
            "mutations_reverted": self.mutations_reverted,
            "cycle_count": self.cycle_count,
            "probe_file_idx": self._probe_file_idx,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "SelfModifyingCausalOrganism":
        actions = list(raw.get("actions") or AUTONOMOUS_ACTIONS)
        obj = cls(actions=actions)
        obj.axis_names = [str(x) for x in (raw.get("axis_names") or BASE_AXIS_NAMES)]
        obj.projection_matrix = np.asarray(raw.get("projection_matrix") or np.eye(10), dtype=np.float64)
        obj.transition_weights = {
            str(k): np.asarray(v, dtype=np.float64)
            for k, v in (raw.get("transition_weights") or {}).items()
        }
        for act in obj.actions:
            d = len(obj.axis_names)
            if act not in obj.transition_weights:
                obj.transition_weights[act] = np.zeros((d, d + 1), dtype=np.float64)
        obj.action_counts = {str(k): int(v) for k, v in (raw.get("action_counts") or {}).items()}
        obj.action_policy_values = {str(k): float(v) for k, v in (raw.get("action_policy_values") or {}).items()}
        obj.action_lr_scales = {str(k): float(v) for k, v in (raw.get("action_lr_scales") or {}).items()}
        obj.residual_history = {
            str(k): [list(map(float, vec)) for vec in v]
            for k, v in (raw.get("residual_history") or {}).items()
        }
        obj.episode_buffer = list(raw.get("episode_buffer") or [])
        obj._pending_verification = {
            str(k): {
                **v,
                "weights_before": np.asarray(v["weights_before"], dtype=np.float64),
            }
            for k, v in (raw.get("pending_verification") or {}).items()
        }
        obj._last_error_by_action = {str(k): float(v) for k, v in (raw.get("last_error_by_action") or {}).items()}
        obj.mutations_kept = int(raw.get("mutations_kept", 0))
        obj.mutations_reverted = int(raw.get("mutations_reverted", 0))
        obj.cycle_count = int(raw.get("cycle_count", 0))
        obj._probe_file_idx = int(raw.get("probe_file_idx", 0))
        return obj

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "SelfModifyingCausalOrganism":
        p = Path(path)
        if not p.is_file():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def project_state(self, raw_vec_10: Sequence[float] | np.ndarray) -> np.ndarray:
        """Maps a raw 10-D sensory measurement into current D-D mutable perceptual space."""
        v10 = np.asarray(raw_vec_10, dtype=np.float64).reshape(10)
        return (self.projection_matrix @ v10).astype(np.float64)

    def predict_next(
        self,
        state_d: np.ndarray,
        action: str,
        weights_override: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Predicts next perceptual state $\hat{S}_{t+1} = S_t + W_a [S_t, 1]$."""
        w_mat = weights_override if weights_override is not None else self.transition_weights[action]
        x_aug = np.concatenate([state_d, [1.0]])
        return (state_d + w_mat @ x_aug).astype(np.float64)

    def choose_autonomous_action(self, state_d: np.ndarray) -> str:
        """Selects an action autonomously from curiosity (1/sqrt(1+n)) + learned policy values."""
        best_act = self.actions[0]
        best_score = -1e12
        for act in self.actions:
            c = self.action_counts.get(act, 0)
            explore = 0.45 / math.sqrt(1.0 + c)
            # Expected informative prediction change magnitude
            w_mat = self.transition_weights.get(act)
            pred_mag = 0.0
            if w_mat is not None:
                delta_pred = self.predict_next(state_d, act) - state_d
                pred_mag = float(min(0.4, np.linalg.norm(delta_pred)))
            val = float(self.action_policy_values.get(act, 0.0))
            score = val + explore + 0.15 * pred_mag
            if score > best_score:
                best_score = score
                best_act = act
        return best_act

    def compare_prediction_and_reality(
        self,
        s_before: np.ndarray,
        action: str,
        predicted: np.ndarray,
        actual: np.ndarray,
    ) -> DiscrepancyContrast:
        """Computes exact 4-axis discrepancy (What / How much / How / Why)."""
        diff = (actual - predicted).astype(np.float64)
        l2 = float(np.linalg.norm(diff))
        actual_shift = float(np.linalg.norm(actual - s_before))
        tol = 0.015

        what_matched: List[str] = []
        what_diverged: List[str] = []
        how_signed: Dict[str, float] = {}
        for idx, ax_name in enumerate(self.axis_names):
            dv = round(float(diff[idx]), 5)
            how_signed[ax_name] = dv
            if abs(dv) <= tol:
                what_matched.append(ax_name)
            else:
                what_diverged.append(ax_name)

        count = self.action_counts.get(action, 0)
        hist = self.residual_history.get(action, [])
        aligned_residual = False
        if count >= 3 and len(hist) >= 3 and l2 > 0.08:
            recent_vecs = [np.asarray(h, dtype=np.float64) for h in hist[-3:] if len(h) == len(diff)]
            if len(recent_vecs) == 3:
                mean_prev = np.mean(recent_vecs, axis=0)
                n1 = float(np.linalg.norm(mean_prev))
                n2 = float(np.linalg.norm(diff))
                if n1 > 1e-6 and n2 > 1e-6:
                    cos_sim = float(np.dot(mean_prev, diff) / (n1 * n2))
                    if cos_sim > 0.65:
                        aligned_residual = True

        if action == "observe_still" and actual_shift > 0.02 and l2 > 0.02:
            why_attr = "external_world_autonomy"
            why_detail = f"아무 행동도 하지 않았지만 외부 세계가 스스로 {actual_shift:.4f}만큼 변했다."
        elif count == 0 and l2 > tol:
            why_attr = "unseen_action_effect"
            why_detail = f"'{action}'을 처음 시도해 예측 모델이 백지 상태여서 오차 {l2:.4f}가 발생했다."
        elif aligned_residual:
            why_attr = "missing_sensory_axis"
            why_detail = f"반복 시도({count}회)에도 같은 방향 잔차(L2={l2:.4f})가 해소되지 않아 새로운 지각 축이 필요하다."
        elif l2 > tol:
            why_attr = "model_parameter_drift"
            why_detail = f"예측 가중치와 실제 변화 사이 오차(L2={l2:.4f})가 남아 파라미터 조정이 필요하다."
        else:
            why_attr = "aligned_with_reality"
            why_detail = f"예측과 현실 오차(L2={l2:.4f})가 허용 범위({tol}) 안에 들어왔다."

        return DiscrepancyContrast(
            what_matched=what_matched,
            what_diverged=what_diverged,
            how_much_l2=round(l2, 5),
            how_signed_deltas=how_signed,
            why_attribution=why_attr,
            why_detail=why_detail,
        )

    def _compute_past_error_for_action(self, action: str, weights_override: Optional[np.ndarray] = None) -> float:
        errs: List[float] = []
        for ep in self.episode_buffer:
            if ep["action"] != action:
                continue
            sb = self.project_state(ep["raw_before"])
            sa = self.project_state(ep["raw_after"])
            pred = self.predict_next(sb, action, weights_override=weights_override)
            errs.append(float(np.linalg.norm(sa - pred)))
        return round(float(np.mean(errs)), 5) if errs else 0.0

    def _expand_perceptual_axis(self, diff_d: np.ndarray, action: str) -> str:
        """Spawns one orthogonal emergent row into `projection_matrix` and expands all transition matrices."""
        recent = [np.asarray(v, dtype=np.float64) for v in self.residual_history.get(action, [])[-4:] if len(v) == len(diff_d)]
        mean_err_d = np.mean([*recent, diff_d], axis=0) if recent else diff_d
        base_dir, _, _, _ = np.linalg.lstsq(self.projection_matrix, mean_err_d, rcond=None)

        # Orthogonalize against prior emergent axes (rows >= 10)
        for row in self.projection_matrix[len(BASE_AXIS_NAMES) :]:
            denom = float(np.dot(row, row))
            if denom > 1e-10:
                base_dir = base_dir - (float(np.dot(base_dir, row)) / denom) * row

        norm_v = float(np.linalg.norm(base_dir))
        if norm_v < 1e-5:
            return ""
        new_row = (base_dir / norm_v).reshape(1, -1)
        d_old = len(self.axis_names)
        new_name = f"emergent_axis_{d_old - len(BASE_AXIS_NAMES) + 1}"
        self.axis_names.append(new_name)
        self.projection_matrix = np.vstack([self.projection_matrix, new_row])

        d_new = len(self.axis_names)
        for act, old_w in list(self.transition_weights.items()):
            new_w = np.zeros((d_new, d_new + 1), dtype=np.float64)
            new_w[:d_old, :d_old] = old_w[:, :d_old]
            new_w[:d_old, -1] = old_w[:, -1]
            self.transition_weights[act] = new_w

        # Pad residual history to D_new cleanly
        for act, rlist in self.residual_history.items():
            for vec in rlist:
                while len(vec) < d_new:
                    vec.append(0.0)
        return new_name

    def self_modify(
        self,
        raw_before: np.ndarray,
        action: str,
        predicted: np.ndarray,
        actual: np.ndarray,
        contrast: DiscrepancyContrast,
    ) -> SelfModificationRecord:
        """
        Verifies prior mutation (rolling back if real error spiked), applies normalized
        causal weight update, spawns new perceptual axis when warranted, re-evaluates past
        episodes retroactively, and updates action policy via counterfactual comparison.
        """
        self.cycle_count += 1
        err_now = float(contrast.how_much_l2)
        pending = self._pending_verification.get(action)

        # 1. Verify prior mutation on this action — automatic rollback if harmful!
        if pending is not None and err_now > max(0.05, float(pending["error_before_update"]) * 1.35 + 0.03):
            old_w = pending["weights_before"]
            cur_w = self.transition_weights[action]
            past_before = self._compute_past_error_for_action(action)
            if old_w.shape == cur_w.shape:
                p_delta = float(np.linalg.norm(old_w - cur_w))
                self.transition_weights[action] = old_w.copy()
            else:
                p_delta = 0.0
            past_after = self._compute_past_error_for_action(action)
            self.action_lr_scales[action] = max(0.05, float(self.action_lr_scales.get(action, 1.0)) * 0.5)
            self.mutations_reverted += 1
            self.action_policy_values[action] = round(float(self.action_policy_values.get(action, 0.0)) - 0.10, 4)
            self._pending_verification.pop(action, None)
            return SelfModificationRecord(
                cycle=self.cycle_count,
                action_taken=action,
                mutation_kind="rollback",
                parameter_delta_norm=round(p_delta, 5),
                past_error_before=past_before,
                past_error_after=past_after,
                counterfactual_best_action=self.choose_autonomous_action(self.project_state(raw_before)),
                kept=False,
            )

        if pending is not None:
            self.mutations_kept += 1

        # 2. Retroactive past error before mutation
        past_before = self._compute_past_error_for_action(action)
        w_prev = self.transition_weights[action].copy()
        s_before_d = self.project_state(raw_before)
        err_vec = (actual - predicted).astype(np.float64)

        # Record residual before potential expansion
        r_hist = self.residual_history.setdefault(action, [])
        r_hist.append([round(float(x), 6) for x in err_vec])
        if len(r_hist) > 12:
            self.residual_history[action] = r_hist[-12:]

        # 3. Online normalized ridge mutation on transition weights
        mut_kind = "steady_alignment"
        new_axis = ""
        if contrast.why_attribution != "aligned_with_reality":
            cnt = self.action_counts.get(action, 0)
            lr = 0.55 * float(self.action_lr_scales.get(action, 1.0)) / (1.0 + 0.08 * cnt)
            x_aug = np.concatenate([s_before_d, [1.0]])
            denom = float(np.dot(x_aug, x_aug) + 0.25)
            delta_w = lr * np.outer(err_vec, x_aug) / denom
            self.transition_weights[action] = w_prev + delta_w
            mut_kind = "weight_update"
            self._pending_verification[action] = {
                "cycle": self.cycle_count,
                "weights_before": w_prev.copy(),
                "error_before_update": err_now,
                "lr_scale": self.action_lr_scales.get(action, 1.0),
            }

            if contrast.why_attribution == "missing_sensory_axis" and len(self.axis_names) < self.MAX_AXES:
                spawned = self._expand_perceptual_axis(err_vec, action)
                if spawned:
                    new_axis = spawned
                    mut_kind = "axis_expansion"
                    # Seed the newly created perceptual row from the same residual step
                    s_exp_b = self.project_state(raw_before)
                    base_err = np.linalg.lstsq(self.projection_matrix[:-1], err_vec, rcond=None)[0]
                    new_ax_err = float(self.projection_matrix[-1] @ base_err)
                    x_exp_aug = np.concatenate([s_exp_b, [1.0]])
                    self.transition_weights[action][-1, :] = (
                        lr * new_ax_err * x_exp_aug / (float(np.dot(x_exp_aug, x_exp_aug)) + 0.25)
                    )

        p_delta = (
            float(np.linalg.norm(self.transition_weights[action][: w_prev.shape[0], : w_prev.shape[1]] - w_prev))
            if self.transition_weights[action].shape[0] >= w_prev.shape[0]
            else 0.0
        )

        # 4. Re-evaluate past episodes through the newly mutated self
        past_after = self._compute_past_error_for_action(action)
        err_reduction = max(-0.2, past_before - past_after) if past_before > 0 else 0.0

        # 5. Counterfactual simulation across actions using updated weights
        s_now_d = self.project_state(raw_before)
        cf_scores: List[Tuple[float, str]] = []
        for cand in self.actions:
            pred_c = self.predict_next(s_now_d, cand)
            info_mag = float(np.linalg.norm(pred_c - s_now_d))
            cf_scores.append((self.action_policy_values.get(cand, 0.0) + 0.25 * info_mag, cand))
        cf_scores.sort(key=lambda item: (-item[0], item[1]))
        cf_best = cf_scores[0][1]

        # Revise policy value based on verified error drop & predictability gain
        prev_err = self._last_error_by_action.get(action)
        step_gain = (prev_err - err_now) if prev_err is not None else 0.0
        self.action_policy_values[action] = round(
            float(np.clip(self.action_policy_values.get(action, 0.0) + 0.25 * err_reduction + 0.15 * step_gain, -1.0, 2.0)),
            4,
        )
        self._last_error_by_action[action] = err_now

        return SelfModificationRecord(
            cycle=self.cycle_count,
            action_taken=action,
            mutation_kind=mut_kind,
            parameter_delta_norm=round(p_delta, 5),
            past_error_before=past_before,
            past_error_after=past_after,
            counterfactual_best_action=cf_best,
            new_axis_name=new_axis,
            kept=True,
        )

    def _execute_embodied_action(
        self,
        action: str,
        retina: ScreenRetinaSensor,
        workspace_root: Optional[Path] = None,
    ) -> np.ndarray:
        """Executes a real non-scripted action in the external world and returns the true 10-D state."""
        if action == "observe_still":
            snap_after = retina.capture()
            return encode_live_world_state(snap_after)

        if action in ("zoom_salient_focus", "zoom_opposite_quadrant"):
            center = None
            if action == "zoom_opposite_quadrant":
                sx, sy = retina._last_salient_xy
                center = (float(np.clip(1.0 - sx, 0.1, 0.9)), float(np.clip(1.0 - sy, 0.1, 0.9)))
            crop = retina.capture_salient_zoom_crop(center_xy_norm=center, zoom_factor=3.0)
            snap_after = retina.capture()
            if crop.get("ok"):
                return encode_live_world_state(
                    snap_after,
                    edge_override=float(np.clip(crop["crop_edge_sharpness"] * 3.0, 0.0, 1.0)),
                    contrast_override=float(crop["crop_contrast_std"]),
                    rgb_override=crop["crop_mean_rgb"],
                    warm_cool_override=float(crop["crop_warm_cool_ratio"]),
                )
            return encode_live_world_state(snap_after)

        if action == "probe_workspace_file" and workspace_root is not None and workspace_root.exists():
            candidates = sorted(
                p for p in workspace_root.iterdir()
                if p.is_file() and p.suffix.lower() in (".md", ".py", ".txt", ".json")
            )
            snap_after = retina.capture()
            if candidates:
                chosen_file = candidates[self._probe_file_idx % len(candidates)]
                self._probe_file_idx += 1
                f_res = inspect_local_file_sensory(chosen_file)
                if f_res.get("ok") and f_res.get("kind") == "text":
                    char_density = float(np.clip(f_res["char_count"] / 3000.0, 0.0, 1.0))
                    line_density = float(np.clip(f_res["line_count"] / 80.0, 0.0, 1.0))
                    return encode_live_world_state(
                        snap_after,
                        edge_override=line_density,
                        contrast_override=char_density,
                    )
            return encode_live_world_state(snap_after)

        return encode_live_world_state(retina.capture())

    def run_autonomous_cycle(
        self,
        retina: ScreenRetinaSensor,
        fg_sensor: WindowsForegroundSensor,
        workspace_root: Optional[Path] = None,
        forced_action: Optional[str] = None,
    ) -> OrganismCycleTrace:
        """Runs one full uncontrolled Sense -> Predict -> Act -> Contrast -> Self-Modify cycle."""
        snap_before = retina.capture()
        fg_title = fg_sensor.get_active_window_title()
        raw_before = encode_live_world_state(snap_before)
        s_before_d = self.project_state(raw_before)

        action = forced_action if (forced_action and forced_action in self.actions) else self.choose_autonomous_action(s_before_d)
        predicted_d = self.predict_next(s_before_d, action)

        raw_after = self._execute_embodied_action(action, retina=retina, workspace_root=workspace_root)
        actual_d = self.project_state(raw_after)

        contrast = self.compare_prediction_and_reality(s_before_d, action, predicted_d, actual_d)
        mod = self.self_modify(raw_before, action, predicted_d, actual_d, contrast)

        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        self.episode_buffer.append(
            {
                "cycle": mod.cycle,
                "action": action,
                "raw_before": [round(float(x), 6) for x in raw_before],
                "raw_after": [round(float(x), 6) for x in raw_after],
                "error_norm": contrast.how_much_l2,
                "why": contrast.why_attribution,
                "mutation": mod.mutation_kind,
            }
        )
        if len(self.episode_buffer) > self.MAX_EPISODES:
            self.episode_buffer = self.episode_buffer[-self.MAX_EPISODES :]

        # Immediate re-observation norm using mutated model on the just-observed transition
        s_before_after_mut = self.project_state(raw_before)
        s_actual_after_mut = self.project_state(raw_after)
        repred = self.predict_next(s_before_after_mut, action)
        reobs_err = round(float(np.linalg.norm(s_actual_after_mut - repred)), 5)

        return OrganismCycleTrace(
            cycle=mod.cycle,
            action_chosen=action,
            predicted_next=[round(float(x), 5) for x in predicted_d],
            actual_next=[round(float(x), 5) for x in actual_d],
            contrast=contrast,
            modification=mod,
            reobserved_error_norm=reobs_err,
            active_axes_count=len(self.axis_names),
            window_title=fg_title,
        )
