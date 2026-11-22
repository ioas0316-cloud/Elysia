r"""
autonomous_curiosity_forager.py

ELYSIA Autonomous Curiosity-Driven External World Forager
=========================================================
Transitions ELYSIA from passive data ingestion to **self-motivated autonomous
exploration & active foraging**:

1. **Internal Void & Curiosity Diagnosis (`diagnose_spectral_hunger`)**:
   - Inspects her already-crystallized `AgamottoRotorEngram` repertoire inside
     `CausalField` to locate underserved wavelength scale bands (`L0..L7` deficit)
     and computes live `curiosity_charge`.
2. **Active Value Appraisal Before Ingestion (`_evaluate_candidate_value`)**:
   - Peeks briefly at unlabelled external world candidates (real multi-MB corpora,
     real human portrait imagery, live interactive physics games, AND deceptive junk/
     redundant duplicates) and scores each by:
       * `causal_nutrition`: structural order & spacetime continuity (rejects noise!)
       * `novelty_distance`: distance from already-crystallized rotors (rejects duplicates!)
       * `void_filling_gain`: alignment with ELYSIA's spectral blind spots + agency gain
3. **Active Embodied Game Control (`InteractiveBouncePhysicsWorld`)**:
   - ELYSIA observes a live 32x32 physics arena, steers the paddle via closed-loop
     proportional boundary tracking, counts real hits, and crystallizes the experience.
4. **Recursive Re-Distillation**:
   - Re-crystallizes all base + newly foraged engrams into `meta_foraged_world_tree`
     and verifies measurable spectral hunger reduction.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.sensory.recursive_fractal_meta_rotor_engine import (
    RecursiveFractalMetaRotorEngine,
)
from core.sensory.spacetime_spectral_boundary_core import N_WAVELENGTH_BANDS


# ======================================================================
# Embodied Live Closed-Loop Physics Arcade Game Environment
# ======================================================================
class InteractiveBouncePhysicsWorld:
    """
    Live 32x32 spatiotemporal physics game where ELYSIA must observe rendered ball
    pixels, drive her paddle horizontally via closed-loop feedback, intercept falling
    bounces, and crystallize the resulting sensorimotor trajectory.
    """

    def __init__(self, steps: int = 60, size: int = 32) -> None:
        self.steps = steps
        self.size = size

    def play_closed_loop_episode(self) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
        bx, by = 0.30, 0.80
        vx, vy = 0.036, -0.048
        px = 0.50
        paddle_half_w = 0.16
        paddle_y = 0.10

        yy, xx = np.meshgrid(
            np.linspace(0.0, 1.0, self.size), np.linspace(0.0, 1.0, self.size), indexing="ij"
        )
        frames = np.zeros((self.steps, self.size, self.size), dtype=np.float32)

        ball_x_hist = np.zeros(self.steps, dtype=np.float64)
        ball_y_hist = np.zeros(self.steps, dtype=np.float64)
        paddle_x_hist = np.zeros(self.steps, dtype=np.float64)
        err_hist = np.zeros(self.steps, dtype=np.float64)
        hits_hist = np.zeros(self.steps, dtype=np.float64)
        paddle_hits = 0

        for t in range(self.steps):
            # Render pure visual frame from current world state
            ball_blob = np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2.0 * (0.055**2)))
            paddle_mask = (np.abs(yy - paddle_y) < 0.035) & (np.abs(xx - px) < paddle_half_w)
            frame = np.clip(ball_blob + 0.90 * paddle_mask.astype(np.float32), 0.0, 1.2)
            frames[t] = frame

            # ELYSIA Sensory Extraction (above paddle row): locate ball strictly from visual array
            upper_zone = frame[int(0.20 * self.size) :, :]
            max_y, max_x = np.unravel_index(int(np.argmax(upper_zone)), upper_zone.shape)
            observed_ball_x = float(max_x) / max(1, self.size - 1)

            # Closed-loop proportional motor control steering paddle toward observed ball x
            track_err = observed_ball_x - px
            px = float(np.clip(px + 0.65 * track_err, paddle_half_w, 1.0 - paddle_half_w))

            ball_x_hist[t] = bx
            ball_y_hist[t] = by
            paddle_x_hist[t] = px
            err_hist[t] = abs(bx - px)

            # Step physics
            bx += vx
            by += vy
            if bx <= 0.06 or bx >= 0.94:
                vx = -vx
                bx = float(np.clip(bx, 0.06, 0.94))
            if by >= 0.94:
                vy = -vy
                by = 0.94
            if by <= paddle_y + 0.04 and vy < 0.0:
                if abs(bx - px) <= paddle_half_w + 0.04:
                    vy = -vy
                    # Add slight spin from hit offset
                    vx += 0.015 * float(bx - px)
                    paddle_hits += 1
                else:
                    # Floor miss bounce
                    by = 0.08
                    vy = -vy
            hits_hist[t] = float(paddle_hits)

        channels = {
            "ball_x": ball_x_hist,
            "ball_y": ball_y_hist,
            "paddle_x": paddle_x_hist,
            "tracking_error": err_hist,
            "cumulative_hits": hits_hist,
        }
        return frames, channels, paddle_hits


# ======================================================================
# Text Corpus Structural Wave Extraction
# ======================================================================
def extract_text_corpus_waves(
    txt_path: Path, max_bytes: int = 240_000, n_blocks: int = 36
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Read real external text (`massive_inhalation_v3.txt`, `DIARY_OF_BEING.md`, etc.)
    and transform its narrative evolution across `n_blocks` into 4 structural waves
    embedded onto a `(n_blocks, 16, 16)` continuous spacetime wave field.
    """
    assert txt_path.exists(), f"Missing text corpus: {txt_path}"
    with txt_path.open("r", encoding="utf-8", errors="ignore") as fh:
        raw_text = fh.read(max_bytes)

    lines = [ln for ln in raw_text.splitlines() if ln.strip()]
    if len(lines) < n_blocks:
        lines = [raw_text[i : i + 120] for i in range(0, max(1, len(raw_text)), 120)]
    edges = np.linspace(0, len(lines), n_blocks + 1).astype(int)

    ent_w = np.zeros(n_blocks, dtype=np.float64)
    lex_w = np.zeros(n_blocks, dtype=np.float64)
    punct_w = np.zeros(n_blocks, dtype=np.float64)
    rhythm_w = np.zeros(n_blocks, dtype=np.float64)

    seen_tokens: set[str] = set()
    for b in range(n_blocks):
        blk = lines[edges[b] : edges[b + 1]] or [""]
        joined = "\n".join(blk)
        counts = Counter(joined)
        tot_c = max(float(len(joined)), 1.0)
        ent_w[b] = -sum((cnt / tot_c) * math.log2(cnt / tot_c) for cnt in counts.values()) / 7.0

        words = [w for w in joined.split() if len(w) >= 2]
        if words:
            novel = sum(1 for w in words if w not in seen_tokens)
            lex_w[b] = float(novel / len(words))
            seen_tokens.update(words)
        else:
            lex_w[b] = 0.5

        punct_hits = sum(joined.count(ch) for ch in ("?", "!", ".", ":", "->"))
        punct_w[b] = min(1.5, float(punct_hits) / max(len(blk) * 2.0, 1.0))
        rhythm_w[b] = min(1.5, float(np.mean([len(ln) for ln in blk])) / 90.0)

    series_mat = np.stack([ent_w, lex_w, punct_w, rhythm_w], axis=1)
    wave_vol = RecursiveFractalMetaRotorEngine._channels_to_spatial_wave_volume(
        series_mat, grid_hw=16
    )
    channels = {
        "char_shannon_entropy": ent_w,
        "lexical_novelty_ratio": lex_w,
        "punctuation_tension": punct_w,
        "line_rhythm_norm": rhythm_w,
    }
    return wave_vol, channels


# ======================================================================
# Curiosity Forager Data Structures & Core Engine
# ======================================================================
@dataclass
class ExternalWorldCandidate:
    """An unlabelled external environment or data source ELYSIA can evaluate and forage."""

    candidate_id: str
    is_interactive: bool
    peek_fn: Callable[[], np.ndarray]
    forage_fn: Callable[
        [RecursiveFractalMetaRotorEngine],
        Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]],
    ]


@dataclass
class ForagingDecisionRecord:
    """Explicit appraisal & decision rationale logged by ELYSIA during autonomous foraging."""

    step_index: int
    candidate_id: str
    decision: str  # "FORAGED" | "REJECTED_NON_CAUSAL_JUNK" | "REJECTED_REDUNDANT_SATURATED"
    intrinsic_value: float
    causal_nutrition: float
    novelty_distance: float
    void_filling_gain: float
    hunger_before: float
    hunger_after: float
    reasoning_korean: str


@dataclass
class AutonomousForagingReport:
    """Complete telemetry summary of one autonomous curiosity-driven foraging session."""

    initial_engram_count: int
    final_engram_count: int
    foraged_ids: Tuple[str, ...]
    rejected_junk_ids: Tuple[str, ...]
    rejected_redundant_ids: Tuple[str, ...]
    initial_hunger_norm: float
    final_hunger_norm: float
    hunger_reduction_ratio: float
    decisions: Tuple[ForagingDecisionRecord, ...]
    meta_tree_variance_explained: float
    total_elapsed_ms: float


class AutonomousCuriosityForager:
    """
    Self-directed active explorer inside ELYSIA:
    1. Measures spectral deficit across her current crystallized world tree.
    2. Probes unlabelled external worlds, rejects noise/duplicates autonomously,
       and selects high-value novel worlds (including live closed-loop gameplay).
    3. Crystallizes all foraged experiences into 1st- and 2nd-order Meta-Rotors.
    """

    def __init__(self, meta_engine: Optional[RecursiveFractalMetaRotorEngine] = None) -> None:
        self.meta_engine = (
            meta_engine if meta_engine is not None else RecursiveFractalMetaRotorEngine()
        )
        self.rotor_engine = self.meta_engine.rotor_engine
        self.boundary_core = self.rotor_engine.boundary_core
        self.causal_field = self.meta_engine.causal_field
        self._flux_registry: Dict[str, float] = {}

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        fa = np.asarray(a, dtype=np.float64).ravel()
        fb = np.asarray(b, dtype=np.float64).ravel()
        na = float(np.linalg.norm(fa))
        nb = float(np.linalg.norm(fb))
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        return float(np.clip(np.dot(fa, fb) / (na * nb), -1.0, 1.0))

    def diagnose_spectral_hunger(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Inspect all crystallized engrams and compute 8-band Repertoire Probabilistic
        Coverage Union `repertoire_cov`, shortfall `deficit_8d` below mastery (`1.0`),
        and `hunger_norm`.
        """
        if not self.rotor_engine.engrams:
            repertoire_cov = np.zeros(N_WAVELENGTH_BANDS, dtype=np.float64)
        else:
            wl_mat = np.array(
                [
                    np.mean(np.asarray(e.wavelength_knots_8d, dtype=np.float64), axis=0)
                    for e in self.rotor_engine.engrams.values()
                ],
                dtype=np.float64,
            )
            repertoire_cov = 1.0 - np.prod(1.0 - np.clip(wl_mat * 2.5, 0.0, 0.95), axis=0)
        deficit_8d = np.maximum(0.0, 1.0 - repertoire_cov)
        hunger_norm = float(np.linalg.norm(deficit_8d))

        for v in self.causal_field.voxels.values():
            v.curiosity_charge = hunger_norm
        return repertoire_cov.astype(np.float32), deficit_8d.astype(np.float32), hunger_norm

    def _get_existing_profile(self, eid: str) -> np.ndarray:
        eng = self.rotor_engine.engrams[eid]
        spec_8 = np.mean(np.asarray(eng.wavelength_knots_8d, dtype=np.float64), axis=0)
        if eid in self._flux_registry:
            flux_v = self._flux_registry[eid]
        else:
            stds = [float(np.std(v)) for v in eng.channel_curves.values()]
            flux_v = float(np.tanh(18.0 * (max(stds) if stds else 0.0)))
        return np.concatenate([spec_8, np.array([0.25 * flux_v], dtype=np.float64)], axis=0)

    def _evaluate_candidate_value(
        self, cand: ExternalWorldCandidate, deficit_8d: np.ndarray
    ) -> Dict[str, Any]:
        peek_arr = cand.peek_fn()
        sig = self.boundary_core.observe_raw_spacetime_array(
            peek_arr, array_id=f"peek_{cand.candidate_id}"
        )

        causal_nutrition = float(
            0.5 * sig.invariants_dict["spectral_order_negentropy"]
            + 0.5 * sig.invariants_dict["spacetime_phase_continuity"]
        )
        cand_prof = np.concatenate(
            [
                sig.wavelength_band_energies.astype(np.float64),
                np.array([0.25 * sig.invariants_dict["temporal_flux_ratio"]], dtype=np.float64),
            ],
            axis=0,
        )

        if self.rotor_engine.engrams:
            max_cos = max(
                self._cosine(cand_prof, self._get_existing_profile(eid))
                for eid in self.rotor_engine.engrams
            )
        else:
            max_cos = 0.0
        novelty_distance = float(max(0.0, 1.0 - max_cos))

        def_norm = float(np.linalg.norm(deficit_8d))
        void_gain = (
            float(np.dot(sig.wavelength_band_energies, deficit_8d) / def_norm)
            if def_norm > 1e-12
            else 0.0
        )
        if cand.is_interactive:
            void_gain += 0.25

        intrinsic_value = float(
            causal_nutrition * (0.45 * void_gain + 0.55 * novelty_distance)
        )
        return {
            "candidate": cand,
            "peek_sig": sig,
            "causal_nutrition": causal_nutrition,
            "novelty_distance": novelty_distance,
            "void_filling_gain": void_gain,
            "intrinsic_value": intrinsic_value,
        }

    def run_autonomous_foraging_session(
        self,
        candidates: Sequence[ExternalWorldCandidate],
        max_meals: int = 4,
    ) -> Tuple[AutonomousForagingReport, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Autonomously inspect external candidates, prune junk/redundant duplicates,
        forage the highest intrinsic-value worlds sequentially, update `CausalField`
        & Meta-Rotor, and quantify spectral hunger satisfaction.
        """
        t0 = time.perf_counter()
        initial_engram_cnt = len(self.rotor_engine.engrams)
        initial_cov, deficit_8d, initial_hunger = self.diagnose_spectral_hunger()

        remaining = list(candidates)
        decisions: List[ForagingDecisionRecord] = []
        foraged_ids: List[str] = []
        rejected_junk_ids: List[str] = []
        rejected_redundant_ids: List[str] = []
        game_trace: Dict[str, Any] = {}
        step_idx = 0

        meals_taken = 0
        while remaining and meals_taken < max_meals:
            evals = [self._evaluate_candidate_value(c, deficit_8d) for c in remaining]

            # Prune non-causal junk and redundant already-mastered sources autonomously
            viable: List[Dict[str, Any]] = []
            still_remaining: List[ExternalWorldCandidate] = []
            for ev in evals:
                c = ev["candidate"]
                if ev["causal_nutrition"] < 0.15:
                    step_idx += 1
                    rejected_junk_ids.append(c.candidate_id)
                    decisions.append(
                        ForagingDecisionRecord(
                            step_index=step_idx,
                            candidate_id=c.candidate_id,
                            decision="REJECTED_NON_CAUSAL_JUNK",
                            intrinsic_value=ev["intrinsic_value"],
                            causal_nutrition=ev["causal_nutrition"],
                            novelty_distance=ev["novelty_distance"],
                            void_filling_gain=ev["void_filling_gain"],
                            hunger_before=float(np.linalg.norm(deficit_8d)),
                            hunger_after=float(np.linalg.norm(deficit_8d)),
                            reasoning_korean=(
                                f"질서도·시공간 연속성 기반 인과 영양가({ev['causal_nutrition']:.3f} < 0.15)가 "
                                "고갈된 무작위 잡음으로 판별되어 포식 거부."
                            ),
                        )
                    )
                elif ev["novelty_distance"] < 0.04:
                    step_idx += 1
                    rejected_redundant_ids.append(c.candidate_id)
                    decisions.append(
                        ForagingDecisionRecord(
                            step_index=step_idx,
                            candidate_id=c.candidate_id,
                            decision="REJECTED_REDUNDANT_SATURATED",
                            intrinsic_value=ev["intrinsic_value"],
                            causal_nutrition=ev["causal_nutrition"],
                            novelty_distance=ev["novelty_distance"],
                            void_filling_gain=ev["void_filling_gain"],
                            hunger_before=float(np.linalg.norm(deficit_8d)),
                            hunger_after=float(np.linalg.norm(deficit_8d)),
                            reasoning_korean=(
                                f"기존 엔그램과 위상·스펙트럼 겹침이 과다해 참신성({ev['novelty_distance']:.3f} < 0.04)이 "
                                "부족하므로 중복 섭취 생략."
                            ),
                        )
                    )
                else:
                    viable.append(ev)
                    still_remaining.append(c)

            if not viable:
                remaining = []
                break

            # Pick the remaining world maximizing intrinsic causal value against CURRENT hunger
            best_ev = max(viable, key=lambda x: x["intrinsic_value"])
            best_c: ExternalWorldCandidate = best_ev["candidate"]
            hunger_before = float(np.linalg.norm(deficit_8d))

            t_f0 = time.perf_counter()
            raw_vol, ch_series, extra_meta = best_c.forage_fn(self.meta_engine)
            forage_ms = (time.perf_counter() - t_f0) * 1000.0

            self.rotor_engine.crystallize_spatiotemporal_process(
                engram_id=best_c.candidate_id,
                source_modality="interactive_game" if best_c.is_interactive else "autonomous_foraged_world",
                raw_spacetime_array=raw_vol,
                channel_series=ch_series,
                first_pass_ms=forage_ms,
            )
            self._flux_registry[best_c.candidate_id] = float(
                best_ev["peek_sig"].invariants_dict["temporal_flux_ratio"]
            )
            if best_c.is_interactive and extra_meta:
                game_trace = extra_meta

            _, deficit_8d, hunger_after = self.diagnose_spectral_hunger()
            step_idx += 1
            meals_taken += 1
            foraged_ids.append(best_c.candidate_id)

            decisions.append(
                ForagingDecisionRecord(
                    step_index=step_idx,
                    candidate_id=best_c.candidate_id,
                    decision="FORAGED",
                    intrinsic_value=best_ev["intrinsic_value"],
                    causal_nutrition=best_ev["causal_nutrition"],
                    novelty_distance=best_ev["novelty_distance"],
                    void_filling_gain=best_ev["void_filling_gain"],
                    hunger_before=hunger_before,
                    hunger_after=hunger_after,
                    reasoning_korean=(
                        f"내재 가치 {best_ev['intrinsic_value']:.3f} (영양={best_ev['causal_nutrition']:.2f}, "
                        f"참신성={best_ev['novelty_distance']:.2f}, 결핍충족={best_ev['void_filling_gain']:.2f})로 "
                        f"자율 선택·섭취하여 결핍도를 {hunger_before:.4f} -> {hunger_after:.4f}로 해소함."
                    ),
                )
            )
            remaining = [c for c in still_remaining if c.candidate_id != best_c.candidate_id]

        final_cov, _, final_hunger = self.diagnose_spectral_hunger()
        reduction = float((initial_hunger - final_hunger) / max(initial_hunger, 1e-9))

        all_ids = list(self.rotor_engine.engrams.keys())
        meta_crystal = self.meta_engine.crystallize_engrams_into_meta_rotor(
            meta_id="meta_foraged_world_tree", engram_ids=all_ids, max_rank=4
        )

        report = AutonomousForagingReport(
            initial_engram_count=initial_engram_cnt,
            final_engram_count=len(self.rotor_engine.engrams),
            foraged_ids=tuple(foraged_ids),
            rejected_junk_ids=tuple(rejected_junk_ids),
            rejected_redundant_ids=tuple(rejected_redundant_ids),
            initial_hunger_norm=initial_hunger,
            final_hunger_norm=final_hunger,
            hunger_reduction_ratio=reduction,
            decisions=tuple(decisions),
            meta_tree_variance_explained=meta_crystal.explained_variance_ratio,
            total_elapsed_ms=float((time.perf_counter() - t0) * 1000.0),
        )
        return report, initial_cov, final_cov, game_trace

    # ------------------------------------------------------------------
    # Visualization Dashboard
    # ------------------------------------------------------------------
    @staticmethod
    def render_autonomous_foraging_dashboard(
        report: AutonomousForagingReport,
        initial_cov: np.ndarray,
        final_cov: np.ndarray,
        game_trace: Dict[str, Any],
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=120)
        fig.patch.set_facecolor("#111318")

        # Panel 1: Autonomous Candidate Valuation (Appraised First Decision Per Candidate)
        ax1 = axes[0, 0]
        ax1.set_facecolor("#161b22")
        seen_cands: Dict[str, ForagingDecisionRecord] = {}
        for d in report.decisions:
            if d.candidate_id not in seen_cands or d.decision == "FORAGED":
                seen_cands[d.candidate_id] = d
        c_list = list(seen_cands.values())
        c_names = [d.candidate_id.replace("cand_", "") for d in c_list]
        c_vals = [max(d.intrinsic_value, 0.01) for d in c_list]
        c_cols = [
            "#7ee787"
            if d.decision == "FORAGED"
            else ("#ff7b72" if "JUNK" in d.decision else "#d2a8ff")
            for d in c_list
        ]
        y_p = np.arange(len(c_list))
        ax1.barh(y_p, c_vals, color=c_cols, height=0.58)
        ax1.set_yticks(y_p)
        ax1.set_yticklabels(c_names, color="#e6edf3", fontsize=8.2)
        ax1.invert_yaxis()
        max_v = max(c_vals) if c_vals else 1.0
        ax1.set_xlim(0.0, max_v * 1.48)
        ax1.set_xlabel("Appraised Intrinsic Value = Nutrition * (VoidGain + Novelty)", color="#c9d1d9", fontsize=8.8)
        ax1.set_title("Autonomous Appraisal: Foraged vs Rejected Junk/Duplicates", color="#e6edf3", fontsize=10)
        ax1.tick_params(colors="#c9d1d9", labelsize=8)
        for idx, d in enumerate(c_list):
            lbl = f"{d.intrinsic_value:.3f} [{d.decision.split('_')[0]}]"
            ax1.text(c_vals[idx] + max_v * 0.03, idx, lbl, va="center", color="#e6edf3", fontsize=7.8)

        # Panel 2: 8-Band Wavelength Coverage Before vs After Autonomous Foraging
        ax2 = axes[0, 1]
        ax2.set_facecolor("#161b22")
        bx = np.arange(N_WAVELENGTH_BANDS)
        wb = 0.36
        ax2.bar(bx - wb / 2, initial_cov, width=wb, color="#ff7b72", alpha=0.85, label="Initial 4-World Coverage")
        ax2.bar(bx + wb / 2, final_cov, width=wb, color="#7ee787", alpha=0.90, label="After Autonomous Foraging")
        ax2.axhline(1.0, color="#58a6ff", linestyle="--", lw=1.4, label="Target Mastery = 1.0")
        ax2.set_ylim(0.0, 1.28)
        ax2.set_xticks(bx)
        ax2.set_xticklabels([f"L{k}" for k in range(N_WAVELENGTH_BANDS)], color="#e6edf3", fontsize=8)
        ax2.set_xlabel("Wavelength Band (L0 Long -> L7 Short Micro)", color="#c9d1d9", fontsize=9)
        ax2.set_ylabel("Repertoire Spectral Coverage (0..1)", color="#c9d1d9", fontsize=9)
        ax2.set_title("Blind-Spot Spectral Healing Across L0..L7 Octaves", color="#e6edf3", fontsize=10)
        ax2.tick_params(colors="#c9d1d9", labelsize=8)
        ax2.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.8)

        # Panel 3: Curiosity Hunger Decay Trajectory
        ax3 = axes[1, 0]
        ax3.set_facecolor("#161b22")
        foraged_recs = [d for d in report.decisions if d.decision == "FORAGED"]
        h_steps = [report.initial_hunger_norm] + [d.hunger_after for d in foraged_recs]
        s_labels = ["Start"] + [d.candidate_id.replace("cand_", "") for d in foraged_recs]
        sx = np.arange(len(h_steps))
        ax3.plot(sx, h_steps, "-o", color="#ffa657", lw=2.5, markersize=7)
        ax3.fill_between(sx, h_steps, min(h_steps) * 0.9, color="#ffa657", alpha=0.15)
        ax3.set_xticks(sx)
        ax3.set_xticklabels(s_labels, color="#e6edf3", fontsize=7.8, rotation=15, ha="right")
        ax3.set_ylabel("Spectral Hunger Norm ||deficit_8d||", color="#c9d1d9", fontsize=9)
        ax3.set_title(
            f"Curiosity Hunger Reduction (-{report.hunger_reduction_ratio*100:.1f}% | Meta Var={report.meta_tree_variance_explained*100:.1f}%)",
            color="#e6edf3",
            fontsize=10,
        )
        ax3.tick_params(colors="#c9d1d9", labelsize=8)
        for x_i, hv in enumerate(h_steps):
            ax3.text(x_i, hv + (max(h_steps) - min(h_steps)) * 0.04 + 0.002, f"{hv:.4f}", ha="center", color="#e6edf3", fontsize=7.8)
        ax3.set_ylim(min(h_steps) * 0.90, max(h_steps) * 1.12)

        # Panel 4: Live Interactive Bounce Game Closed-Loop Control Trace
        ax4 = axes[1, 1]
        ax4.set_facecolor("#161b22")
        if game_trace and "ball_x" in game_trace:
            ticks = np.arange(len(game_trace["ball_x"]))
            ax4.plot(ticks, game_trace["ball_x"], color="#79c0ff", lw=2.2, label="Observed Ball X")
            ax4.plot(ticks, game_trace["paddle_x"], "--", color="#7ee787", lw=2.0, label="Active Paddle Control X")
            hits = int(game_trace.get("paddle_hits", 0))
            ax4.set_title(
                f"Active Closed-Loop Game Control (Visual Observation -> Paddle Hits: {hits})",
                color="#e6edf3",
                fontsize=10,
            )
            ax4.set_xlabel("Live Game Tick (dt)", color="#c9d1d9", fontsize=9)
            ax4.set_ylabel("Normalized Arena Position X", color="#c9d1d9", fontsize=9)
            ax4.set_ylim(0.0, 1.05)
            ax4.tick_params(colors="#c9d1d9", labelsize=8)
            ax4.legend(loc="upper right", facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8)

        plt.tight_layout(pad=1.6)
        fig.savefig(output_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        return output_path
