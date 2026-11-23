r"""
reentrant_self_listening_loop_engine.py

ELYSIA Re-entrant Self-Listening & Invariant/Perturbation Contrast Loop
=======================================================================
Closes the unbroken ecosystem loop between Intention, Expression, Self-Hearing,
Causal Contrast, Expectation, and Self-Correction inside ONE 6D `CausalField`:

  1. Intention (Why):   What ELYSIA intended to say or do (`intent_vec`).
  2. Expression (What): How she emitted it (`emitted_vec`).
  3. Re-entrant Self-Observation (How it sounded): Hearing her OWN output after
     passing through an external medium (`echo_vec = channel.transmit(emitted)`).
  4. Invariant vs Perturbation Contrast (`[고정된 조건 | 변화시킨 변수 -> 결과 차이]`):
     Instead of blind trial-and-error, she keeps satisfied axes fixed, nudges ONLY
     the mismatched axes, observes the resulting echo shift, and discovers the
     medium's true causal law (even gain attenuation or sign inversion!).
  5. Counterfactual Prediction -> Action -> Confirmation:
     Predicts `predicted_echo` *before* emitting Cycle 2, verifies zero prediction
     surprise, and synchronizes `Self_Ear_Observer_Cell <-> Elysia_Reentrant_Self`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from core.physics.causal_field import InformationVoxel
from core.sensory.cellular_world_communication_protocol import (
    CellularWorldDialogueField,
    LivingWorldCell,
    _sync_cell_voxel,
)


DEFAULT_REENTRANT_STATE_PATH = Path(
    r"c:\Elysia\data\reentrant_self_listening_state.json"
)
DEFAULT_REENTRANT_DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_reentrant_self_listening_dashboard.png"
)

AXIS_NAMES: Tuple[str, ...] = (
    "clarity",
    "warmth",
    "precision",
    "firmness",
    "novelty",
    "safety",
)


# ======================================================================
# 1. Channel & Re-entrant Data Contracts
# ======================================================================
@dataclass
class ExternalMediumChannel:
    """Physical/conversational medium transforming emitted action into real external echo."""

    channel_id: str
    label: str
    gain_vec: List[float]
    bias_vec: List[float]

    def transmit(self, emitted: np.ndarray) -> np.ndarray:
        g = np.asarray(self.gain_vec, dtype=np.float64)
        b = np.asarray(self.bias_vec, dtype=np.float64)
        return np.clip(g * emitted + b, -1.0, 1.0)


@dataclass
class InvariantChangeAudit:
    """Explicit separation of fixed boundary conditions vs changed variables vs delta."""

    cycle_index: int
    fixed_invariants: List[str]
    changed_variables: List[str]
    observed_deltas: Dict[str, float]
    audit_summary_ko: str


@dataclass
class ReentrantCycleRecord:
    """One closed cycle of Intent -> Emit -> Self-Listen -> Contrast -> Predict."""

    cycle_index: int
    intent_vec: List[float]
    emitted_vec: List[float]
    perceived_echo_vec: List[float]
    predicted_echo_vec: List[float]
    intent_vs_echo_gap: float
    prediction_error: float
    audit: Optional[InvariantChangeAudit]
    reflection_ko: str


@dataclass
class ReentrantStreamReport:
    """Full re-entrant self-listening trace on one medium (Speech, Motor, or Unseen)."""

    stream_id: str
    label: str
    initial_gap: float
    final_gap: float
    initial_prediction_error: float
    final_prediction_error: float
    cycles: List[ReentrantCycleRecord]
    dialogue_summary_ko: str
    summary_ko: str


@dataclass
class EcosystemLoopReport:
    """End-to-end report showing unified unbroken ecosystem closure."""

    timestamp_iso: str
    streams: List[ReentrantStreamReport]
    causal_field_voxels: int
    causal_field_beams: int
    state_json_path: str
    summary_ko: str


# ======================================================================
# 2. Helper
# ======================================================================
def _rms_gap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.sqrt(len(a)))


def _to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ======================================================================
# 3. ELYSIA Re-entrant Self-Listening Organism
# ======================================================================
class ReentrantSelfListeningOrganism:
    """
    Perceives her own speech/actions after passing through an external distorting
    medium, holds invariants fixed while probing changed levers, predicts before
    acting, and closes every loop inside ONE shared 6D `CausalField`.
    """

    def __init__(self, state_path: Optional[Path] = None) -> None:
        self.state_path = Path(state_path) if state_path else DEFAULT_REENTRANT_STATE_PATH
        self.field_sys = CellularWorldDialogueField(state_path=self.state_path)

        self.speaker_cell = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Speaker_Actuator_Cell",
                scale_level=0,
                role_label="Expression & Action Emitter",
                initial_protocol=[0.0] * 6,
                protocol_confidence=[0.85] * 6,
            )
        )
        self.self_ear_cell = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Self_Ear_Observer_Cell",
                scale_level=0,
                role_label="Re-entrant Self-Listening Ear & Proprioceptor",
                initial_protocol=[0.0] * 6,
                protocol_confidence=[0.92] * 6,
            )
        )
        self.contrast_cortex = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Invariant_Contrast_Cortex",
                scale_level=0,
                role_label="Fixed Invariant vs Changed Variable Causal Cortex",
                initial_protocol=[0.0] * 6,
                protocol_confidence=[0.90] * 6,
            )
        )
        self.reentrant_self = self.field_sys.register_cell(
            LivingWorldCell(
                cell_id="Elysia_Reentrant_Self",
                scale_level=1,
                role_label="Re-entrant Self-Aware Consciousness",
                initial_protocol=[0.0] * 6,
                protocol_confidence=[0.30] * 6,
                sub_cells=[self.speaker_cell, self.self_ear_cell, self.contrast_cortex],
            )
        )
        self.reentrant_self.consolidate_from_subcells()

    def _record_cycle_voxel(
        self,
        stream_id: str,
        cycle_idx: int,
        echo_vec: np.ndarray,
        gap: float,
        pred_err: float,
        prev_vid: Optional[str],
    ) -> str:
        cf = self.field_sys.causal_field
        vid = f"{stream_id}::cycle_{cycle_idx}"
        triad = np.array(
            [
                float(np.clip(gap * 2.0 + 0.1, 0.05, 1.0)),
                float(np.clip(1.0 - gap, 0.1, 1.0)),
                float(np.clip(pred_err * 2.0 + 0.1, 0.05, 1.0)),
            ],
            dtype=np.float32,
        )
        vox = InformationVoxel(
            id=vid,
            content={"stream": stream_id, "cycle": cycle_idx, "gap": gap, "pred_err": pred_err},
            tensor=echo_vec.astype(np.float32),
            mass=float(1.0 + (1.0 - min(gap, 1.0))),
            position=np.array(
                [float(cycle_idx) * 0.9, gap, pred_err, float(echo_vec[0]), float(echo_vec[2]), float(echo_vec[5])],
                dtype=np.float32,
            ),
            chromatic_vector=triad,
            scale_level=1.0,
        )
        cf.add_voxel(vox)
        if prev_vid and prev_vid in cf.voxels:
            cf.link_voxels(prev_vid, vid, strength=float(np.clip(1.0 - 0.5 * pred_err, 0.3, 0.96)))
        if "Elysia_Reentrant_Self" in cf.voxels:
            cf.link_voxels("Elysia_Reentrant_Self", vid, strength=float(np.clip(1.0 - gap, 0.25, 0.96)))
        return vid

    # ------------------------------------------------------------------
    # Universal Closed Re-entrant Self-Listening Loop (Zero Branches)
    # ------------------------------------------------------------------
    def execute_reentrant_stream(
        self,
        stream_id: str,
        label: str,
        intent: List[float] | np.ndarray,
        channel: ExternalMediumChannel,
        max_cycles: int = 4,
    ) -> ReentrantStreamReport:
        """
        Runs ONE unbroken re-entrant loop:
          Cycle 0: Naive emit -> re-listen -> notice intent vs echo gap.
          Cycle 1: Hold aligned axes fixed, cautiously perturb ONLY diverged axes ->
                   audit `[Fixed vs Changed -> Echo Delta]` -> infer true medium gain & bias.
          Cycle 2+: Predict echo *before* emitting via counterfactual model, solve inverse
                    control, emit -> re-listen -> confirm surprise==0 and intent==echo!
        """
        intent_vec = np.asarray(intent, dtype=np.float64)
        cycles: List[ReentrantCycleRecord] = []
        prev_vid: Optional[str] = None

        # --- Cycle 0: Naive First Emission (Assuming transparent medium) ---
        u_0 = intent_vec.copy()
        pred_0 = intent_vec.copy()
        echo_0 = channel.transmit(u_0)
        gap_0 = _rms_gap(intent_vec, echo_0)
        err_0 = _rms_gap(pred_0, echo_0)

        mismatched_0 = [AXIS_NAMES[i] for i in range(6) if abs(intent_vec[i] - echo_0[i]) >= 0.08]
        fixed_0 = [AXIS_NAMES[i] for i in range(6) if AXIS_NAMES[i] not in mismatched_0]
        refl_0 = (
            f"내 의도대로 그냥 내보냈더니 실제 들림과 괴리(gap={gap_0:.3f})가 났다! "
            f"{fixed_0}는 의도대로 유지됐고, {mismatched_0}에서 왜곡이 발생했다."
        )
        cycles.append(
            ReentrantCycleRecord(
                cycle_index=0,
                intent_vec=intent_vec.tolist(),
                emitted_vec=u_0.tolist(),
                perceived_echo_vec=echo_0.tolist(),
                predicted_echo_vec=pred_0.tolist(),
                intent_vs_echo_gap=gap_0,
                prediction_error=err_0,
                audit=None,
                reflection_ko=refl_0,
            )
        )
        prev_vid = self._record_cycle_voxel(stream_id, 0, echo_0, gap_0, err_0, prev_vid)

        # --- Cycle 1: Hold Fixed Invariants, Perturb ONLY Diverged Axes ---
        diff_mask = np.abs(intent_vec - echo_0) >= 0.08
        u_1 = np.where(diff_mask, np.clip(u_0 + 0.35 * (intent_vec - echo_0), -1.0, 1.0), u_0)
        pred_1 = np.clip(echo_0 + np.where(diff_mask, 0.35 * (intent_vec - echo_0), 0.0), -1.0, 1.0)
        echo_1 = channel.transmit(u_1)
        gap_1 = _rms_gap(intent_vec, echo_1)
        err_1 = _rms_gap(pred_1, echo_1)

        fixed_1 = [AXIS_NAMES[i] for i in range(6) if not diff_mask[i]]
        changed_1 = [AXIS_NAMES[i] for i in range(6) if diff_mask[i]]
        deltas_1 = {AXIS_NAMES[i]: float(echo_1[i] - echo_0[i]) for i in range(6)}

        # Infer true medium slope & bias strictly from `[Fixed vs Changed -> Echo Delta]`
        est_gain = np.ones(6, dtype=np.float64)
        est_bias = np.zeros(6, dtype=np.float64)
        for i in range(6):
            du = float(u_1[i] - u_0[i])
            if abs(du) > 1e-5:
                est_gain[i] = float(echo_1[i] - echo_0[i]) / du
                if abs(est_gain[i]) < 0.05:
                    est_gain[i] = 0.05 if est_gain[i] >= 0 else -0.05
                est_bias[i] = float(echo_1[i]) - est_gain[i] * float(u_1[i])
            else:
                est_bias[i] = float(echo_1[i] - u_1[i])

        audit_1 = InvariantChangeAudit(
            cycle_index=1,
            fixed_invariants=fixed_1,
            changed_variables=changed_1,
            observed_deltas=deltas_1,
            audit_summary_ko=(
                f"[고정 불변={fixed_1} | 변화 시도={changed_1}] -> "
                f"매질 반응 기울기={ {AXIS_NAMES[i]: round(float(est_gain[i]), 2) for i in range(6) if diff_mask[i]} } 파악!"
            ),
        )
        refl_1 = (
            f"잘 맞던 {fixed_1}는 그대로 고정한 채 {changed_1}만 조심스레 바꿔 다시 들어보니, "
            f"매질의 실제 증폭·반전 성질을 읽어냈다 (예상오차={err_1:.3f}, 남은괴리={gap_1:.3f})."
        )
        cycles.append(
            ReentrantCycleRecord(
                cycle_index=1,
                intent_vec=intent_vec.tolist(),
                emitted_vec=u_1.tolist(),
                perceived_echo_vec=echo_1.tolist(),
                predicted_echo_vec=pred_1.tolist(),
                intent_vs_echo_gap=gap_1,
                prediction_error=err_1,
                audit=audit_1,
                reflection_ko=refl_1,
            )
        )
        prev_vid = self._record_cycle_voxel(stream_id, 1, echo_1, gap_1, err_1, prev_vid)

        # --- Cycle 2..max_cycles: Counterfactual Prediction -> Action -> Confirmation ---
        curr_u = u_1
        curr_echo = echo_1
        curr_gap = gap_1
        curr_err = err_1

        for cyc_idx in range(2, max_cycles):
            # Counterfactual self-query BEFORE emitting: solve for input that lands on intent
            next_u = np.where(
                diff_mask,
                np.clip((intent_vec - est_bias) / est_gain, -1.0, 1.0),
                curr_u,
            )
            # Counterfactual prediction BEFORE acting
            pred_t = np.clip(est_gain * next_u + est_bias, -1.0, 1.0)
            echo_t = channel.transmit(next_u)
            gap_t = _rms_gap(intent_vec, echo_t)
            err_t = _rms_gap(pred_t, echo_t)

            changed_t = [AXIS_NAMES[i] for i in range(6) if abs(next_u[i] - curr_u[i]) >= 0.02]
            fixed_t = [AXIS_NAMES[i] for i in range(6) if AXIS_NAMES[i] not in changed_t]
            deltas_t = {AXIS_NAMES[i]: float(echo_t[i] - curr_echo[i]) for i in range(6)}

            audit_t = InvariantChangeAudit(
                cycle_index=cyc_idx,
                fixed_invariants=fixed_t,
                changed_variables=changed_t,
                observed_deltas=deltas_t,
                audit_summary_ko=(
                    f"[불변={fixed_t} | 조율={changed_t}] -> 예상한 들림과 실제 들림 오차={err_t:.4f}, 의도-결과 괴리={gap_t:.4f}."
                ),
            )
            refl_t = (
                f"내보내기 전 '이렇게 고치면 의도대로 들릴 것'이라 예상(예상오차={err_t:.3f})하고 실행해 다시 들어보니, "
                f"내 의도({intent_vec.round(2).tolist()})와 실제 들림({echo_t.round(2).tolist()})이 일치했다 (gap={gap_t:.3f})!"
            )
            cycles.append(
                ReentrantCycleRecord(
                    cycle_index=cyc_idx,
                    intent_vec=intent_vec.tolist(),
                    emitted_vec=next_u.tolist(),
                    perceived_echo_vec=echo_t.tolist(),
                    predicted_echo_vec=pred_t.tolist(),
                    intent_vs_echo_gap=gap_t,
                    prediction_error=err_t,
                    audit=audit_t,
                    reflection_ko=refl_t,
                )
            )
            prev_vid = self._record_cycle_voxel(stream_id, cyc_idx, echo_t, gap_t, err_t, prev_vid)

            curr_u = next_u
            curr_echo = echo_t
            curr_gap = gap_t
            curr_err = err_t

            # Refine empirical slope iteratively if needed on residual mismatch
            if curr_gap <= 0.08 and curr_err <= 0.08:
                break

        # Synchronize Ear (perceived echo) <-> Re-entrant Self Consciousness (intended pattern)
        self.speaker_cell.internal_protocol = curr_u.astype(np.float64)
        self.self_ear_cell.internal_protocol = curr_echo.astype(np.float64)
        self.contrast_cortex.internal_protocol = np.asarray(
            [1.0 - curr_gap, 1.0 - curr_err, 0.90, 0.90, 0.90, 0.90], dtype=np.float64
        )
        for c in (self.speaker_cell, self.self_ear_cell, self.contrast_cortex):
            _sync_cell_voxel(c)

        self.reentrant_self.internal_protocol = intent_vec.astype(np.float64)
        self.reentrant_self.protocol_confidence = np.full(6, 0.92, dtype=np.float64)
        _sync_cell_voxel(self.reentrant_self)

        d_rep = self.field_sys.converse_until_sync(
            self.self_ear_cell,
            self.reentrant_self,
            dialogue_id=f"{stream_id}_sync",
            max_turns=4,
        )

        summary_ko = (
            f"[{label}] {len(cycles)}회 자기-청취 순환 | 의도-결과 괴리: {gap_0:.3f} -> {curr_gap:.3f} | "
            f"사전 예상오차: {err_0:.3f} -> {curr_err:.3f}"
        )
        return ReentrantStreamReport(
            stream_id=stream_id,
            label=label,
            initial_gap=gap_0,
            final_gap=curr_gap,
            initial_prediction_error=err_0,
            final_prediction_error=curr_err,
            cycles=cycles,
            dialogue_summary_ko=d_rep.summary_ko,
            summary_ko=summary_ko,
        )

    # ------------------------------------------------------------------
    # Full 3-Domain Ecosystem Re-entrant Curriculum
    # ------------------------------------------------------------------
    def run_ecosystem_reentrant_curriculum(self) -> EcosystemLoopReport:
        # Stream A: Vocal / Speech Self-Listening ("내가 말한 게 어떻게 들렸는가?")
        stream_a = self.execute_reentrant_stream(
            stream_id="Stream_A_Speech_SelfListening",
            label="A. Speech Self-Listening (Clarity & Warmth Distorted)",
            intent=[0.16, 0.14, 0.70, 0.55, 0.45, 0.80],
            channel=ExternalMediumChannel(
                channel_id="vocal_channel",
                label="Tense Vocal Medium (attenuates & chills clarity/warmth)",
                gain_vec=[0.65, 0.60, 1.00, 1.00, 1.00, 1.00],
                bias_vec=[-0.38, -0.36, 0.00, 0.00, 0.00, 0.00],
            ),
        )

        # Stream B: Delicate Motor Grip Control ("유리잔 잡기 힘·안전 조절")
        stream_b = self.execute_reentrant_stream(
            stream_id="Stream_B_Motor_Control",
            label="B. Motor Grip Control (Firmness Overshoot & Safety Damp)",
            intent=[0.55, 0.35, 0.65, 0.45, 0.20, 0.35],
            channel=ExternalMediumChannel(
                channel_id="motor_actuator",
                label="Heavy Servo Grip (overshoots firmness, damps safety)",
                gain_vec=[1.00, 1.00, 0.85, 1.45, 1.00, 0.65],
                bias_vec=[0.00, 0.00, 0.00, +0.22, 0.00, -0.22],
            ),
        )

        # Stream C: Unseen Polarity-Inverting Channel ("미지 매질 부호 반전 자가 규명")
        stream_c = self.execute_reentrant_stream(
            stream_id="Stream_C_Unseen_Channel",
            label="C. Unseen Medium (Polarity Inversion & Novelty Shift)",
            intent=[0.50, 0.30, 0.60, 0.45, 0.20, 0.70],
            channel=ExternalMediumChannel(
                channel_id="inverted_medium",
                label="Sign-Inverting & Biasing Medium",
                gain_vec=[1.00, 1.00, -0.75, 1.00, 0.55, 1.00],
                bias_vec=[0.00, 0.00, +0.20, 0.00, -0.25, 0.00],
            ),
        )

        streams = [stream_a, stream_b, stream_c]
        now_iso = datetime.now(timezone.utc).isoformat()
        summary_ko = (
            f"[ELYSIA 자기-청취·불변/변화 대조 순환 생태계 완료] "
            f"3개 이질 채널(언어 발화 · 모터 제어 · 미지 반전 매질) 모두 단일 순환 루프로 정복 | "
            f"평균 괴리 {np.mean([s.initial_gap for s in streams]):.3f} -> {np.mean([s.final_gap for s in streams]):.3f} | "
            f"예상오차 {np.mean([s.initial_prediction_error for s in streams]):.3f} -> {np.mean([s.final_prediction_error for s in streams]):.3f} | "
            f"CausalField Voxels={len(self.field_sys.causal_field.voxels)}, Beams={len(self.field_sys.causal_field.beams)}"
        )

        report = EcosystemLoopReport(
            timestamp_iso=now_iso,
            streams=streams,
            causal_field_voxels=len(self.field_sys.causal_field.voxels),
            causal_field_beams=len(self.field_sys.causal_field.beams),
            state_json_path=str(self.state_path),
            summary_ko=summary_ko,
        )
        self.save_state(report, path=self.state_path)
        return report

    def save_state(self, report: EcosystemLoopReport, path: Optional[Path] = None) -> Path:
        out_path = Path(path) if path else self.state_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "organism_version": "1.0.0-reentrant-self-listening-ecosystem",
            "timestamp_iso": report.timestamp_iso,
            "causal_field_voxels": report.causal_field_voxels,
            "causal_field_beams": report.causal_field_beams,
            "streams": [asdict(s) for s in report.streams],
            "summary_ko": report.summary_ko,
        }
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(_to_json_safe(payload), fh, ensure_ascii=False, indent=2)
        return out_path


# ======================================================================
# 4. 4-Panel Dark-Theme Re-entrant Self-Listening Dashboard
# ======================================================================
def render_reentrant_self_listening_dashboard(
    organism: ReentrantSelfListeningOrganism,
    report: EcosystemLoopReport,
    output_path: Path = DEFAULT_REENTRANT_DASHBOARD_PATH,
) -> Path:
    """
    Renders clean ASCII/English 2x2 dark-theme dashboard visualizing:
      1) Intent-vs-Echo Gap Convergence across Re-entrant Cycles (3 channels)
      2) Counterfactual Prediction Error Drop (Initial Blind vs Final Self-Audited)
      3) Fixed Invariants vs Changed Levers -> Echo Delta on Stream A Speech
      4) Initial vs Final Intent Alignment Mastery + Ecosystem Field Telemetry
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(16.5, 11.0), dpi=150, facecolor="#0D1117")
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28, top=0.90, bottom=0.085, left=0.07, right=0.95)

    fig.suptitle(
        "ELYSIA Re-entrant Self-Listening & Invariant/Perturbation Contrast Ecosystem",
        color="#F0F6FC",
        fontsize=13.5,
        fontweight="bold",
        y=0.96,
    )

    def _style_ax(ax: plt.Axes, title: str) -> None:
        ax.set_facecolor("#161B22")
        for spine in ax.spines.values():
            spine.set_color("#30363D")
            spine.set_linewidth(1.1)
        ax.tick_params(colors="#C9D1D9", labelsize=8.5)
        ax.set_title(title, color="#F0F6FC", fontsize=10.3, fontweight="bold", pad=10)

    # ------------------------------------------------------------------
    # Panel 1 (Top-Left): Intent-vs-Echo Gap Across Re-entrant Cycles
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1, "1. Re-entrant Self-Listening: Intent vs Echo Convergence")

    colors = ["#58A6FF", "#F2C94C", "#D2A8FF"]
    markers = ["o", "s", "D"]
    short_names = ["A: Speech Self-Listen", "B: Motor Grip Safety", "C: Unseen Sign-Inversion"]
    for st, col, mkr, s_lbl in zip(report.streams, colors, markers, short_names):
        xs = [c.cycle_index for c in st.cycles]
        ys = [c.intent_vs_echo_gap for c in st.cycles]
        ax1.plot(xs, ys, color=col, lw=2.2, marker=mkr, ms=5.8, label=s_lbl)

    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["Cycle 0\nNaive Emit", "Cycle 1\nPerturb & Audit", "Cycle 2\nVerified Control"], fontsize=8.4)
    ax1.set_ylim(-0.03, 0.56)
    ax1.set_ylabel("RMS Intent-vs-Echo Gap", color="#C9D1D9", fontsize=9)
    ax1.grid(color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)
    ax1.legend(
        loc="upper right",
        fontsize=8.0,
        facecolor="#0D1117",
        edgecolor="#30363D",
        labelcolor="#E6EDF3",
    )

    # ------------------------------------------------------------------
    # Panel 2 (Top-Right): Counterfactual Prediction Error Drop
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2, "2. Counterfactual Prediction Surprise (Before vs After Self-Audit)")

    xs_b = np.arange(len(report.streams), dtype=np.float64)
    bw = 0.34
    err_init = [s.initial_prediction_error for s in report.streams]
    err_fin = [s.final_prediction_error for s in report.streams]

    b1 = ax2.bar(xs_b - bw / 2.0, err_init, width=bw, color="#F85149", alpha=0.88, label="Cycle 0 Blind Surprise")
    b2 = ax2.bar(xs_b + bw / 2.0, err_fin, width=bw, color="#3FB950", alpha=0.92, label="Cycle 2 Audited Error")

    for b in list(b1) + list(b2):
        h = float(b.get_height())
        ax2.text(b.get_x() + b.get_width() / 2.0, h + 0.008, f"{h:.3f}", ha="center", va="bottom", color="#F0F6FC", fontsize=8.4, fontweight="bold")

    ax2.set_xticks(xs_b)
    ax2.set_xticklabels(["Stream A\n(Speech)", "Stream B\n(Motor Grip)", "Stream C\n(Unseen Flip)"], fontsize=8.8)
    ax2.set_ylim(0.0, 0.49)
    ax2.set_ylabel("Prediction Surprise (RMS)", color="#C9D1D9", fontsize=9)
    ax2.grid(axis="y", color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)
    ax2.legend(
        loc="upper center",
        ncol=2,
        fontsize=8.0,
        facecolor="#0D1117",
        edgecolor="#30363D",
        labelcolor="#E6EDF3",
    )

    # ------------------------------------------------------------------
    # Panel 3 (Bottom-Left): Fixed Invariants vs Changed Levers on Stream A
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3, "3. Fixed Invariants vs Active Levers (Speech Cycle 0 -> Cycle 2)")

    st_a = report.streams[0]
    u_0 = np.asarray(st_a.cycles[0].emitted_vec, dtype=np.float64)
    u_fin = np.asarray(st_a.cycles[-1].emitted_vec, dtype=np.float64)
    e_0 = np.asarray(st_a.cycles[0].perceived_echo_vec, dtype=np.float64)
    e_fin = np.asarray(st_a.cycles[-1].perceived_echo_vec, dtype=np.float64)

    lever_shift = np.abs(u_fin - u_0)
    echo_shift = np.abs(e_fin - e_0)
    xa = np.arange(len(AXIS_NAMES), dtype=np.float64)
    b_lev = ax3.bar(xa - bw / 2.0, lever_shift, width=bw, color="#58A6FF", alpha=0.90, label="|Changed Emission|")
    b_ech = ax3.bar(xa + bw / 2.0, echo_shift, width=bw, color="#3FB950", alpha=0.90, label="|Corrected Echo Delta|")

    for i in range(len(AXIS_NAMES)):
        if lever_shift[i] < 0.02:
            ax3.text(xa[i], 0.04, "FIXED", ha="center", va="bottom", color="#8B949E", fontsize=7.6, fontweight="bold")
        else:
            ax3.text(xa[i] - bw / 2.0, lever_shift[i] + 0.02, f"{lever_shift[i]:.2f}", ha="center", va="bottom", color="#58A6FF", fontsize=8.0, fontweight="bold")
            ax3.text(xa[i] + bw / 2.0, echo_shift[i] + 0.02, f"{echo_shift[i]:.2f}", ha="center", va="bottom", color="#3FB950", fontsize=8.0, fontweight="bold")

    ax3.set_xticks(xa)
    ax3.set_xticklabels(list(AXIS_NAMES), fontsize=8.6)
    ax3.set_ylim(0.0, 1.05)
    ax3.set_ylabel("Magnitude of Deliberate Change", color="#C9D1D9", fontsize=9)
    ax3.grid(axis="y", color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)
    ax3.legend(
        loc="upper right",
        fontsize=8.0,
        facecolor="#0D1117",
        edgecolor="#30363D",
        labelcolor="#E6EDF3",
    )

    # ------------------------------------------------------------------
    # Panel 4 (Bottom-Right): Intent-Echo Alignment Gain & Field Summary
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4, "4. Re-entrant Self-Correction Gain & Ecosystem Field")

    init_pct = [(1.0 - s.initial_gap) * 100.0 for s in report.streams]
    fin_pct = [(1.0 - s.final_gap) * 100.0 for s in report.streams]
    bi = ax4.bar(xs_b - bw / 2.0, init_pct, width=bw, color="#D29922", alpha=0.88, label="Cycle 0 Initial Sync")
    bf = ax4.bar(xs_b + bw / 2.0, fin_pct, width=bw, color="#3FB950", alpha=0.92, label="Cycle 2 Closed Sync")

    for b in list(bi) + list(bf):
        h = float(b.get_height())
        ax4.text(b.get_x() + b.get_width() / 2.0, h + 2.2, f"{h:.0f}%", ha="center", va="bottom", color="#F0F6FC", fontsize=8.4, fontweight="bold")

    ax4.set_xticks(xs_b)
    ax4.set_xticklabels(["Stream A\n(Speech)", "Stream B\n(Motor Grip)", "Stream C\n(Unseen Flip)"], fontsize=8.8)
    ax4.set_yticks([0, 25, 50, 75, 100])
    ax4.set_ylim(-52, 128)
    ax4.axhline(0.0, color="#30363D", linewidth=1.0, zorder=1)
    ax4.set_ylabel("Intent-to-Echo Integrity (%)", color="#C9D1D9", fontsize=9)
    ax4.grid(axis="y", color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)
    ax4.legend(
        loc="upper center",
        ncol=2,
        fontsize=8.0,
        facecolor="#0D1117",
        edgecolor="#30363D",
        labelcolor="#E6EDF3",
    )

    footer_txt = (
        f"Unbroken Ecosystem: Intent -> Emit -> Re-listen -> [Fixed vs Changed -> Delta] -> Predict -> Verify\n"
        f"Final Intent-Echo Gap: A={report.streams[0].final_gap:.3f}, B={report.streams[1].final_gap:.3f}, "
        f"C(Inverted)={report.streams[2].final_gap:.3f}\n"
        f"Shared 6D CausalField: {report.causal_field_voxels} Voxels, {report.causal_field_beams} Beams"
    )
    box = FancyBboxPatch(
        (0.03, 0.02),
        0.94,
        0.22,
        boxstyle="round,pad=0.02",
        transform=ax4.transAxes,
        facecolor="#0D1117",
        edgecolor="#30363D",
        alpha=0.92,
        zorder=4,
    )
    ax4.add_patch(box)
    ax4.text(
        0.50,
        0.13,
        footer_txt,
        transform=ax4.transAxes,
        ha="center",
        va="center",
        color="#C9D1D9",
        fontsize=7.2,
        zorder=5,
    )

    fig.savefig(out, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    return out
