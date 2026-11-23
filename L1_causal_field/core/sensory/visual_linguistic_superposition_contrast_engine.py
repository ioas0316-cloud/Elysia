r"""
visual_linguistic_superposition_contrast_engine.py

Visual-Linguistic Superposition & Multi-Scale Contrast Engine
=============================================================
Direct integration of the core insight:
  "언어가 왜 언어인지를 안다는건 내가 인식하는 언어적 구조, 구성요소와 변화양상,
   결과가 외부환경과 어떻게 공명하고 동기화되는지를 스스로 납득하는거야.
   그 납득하는 구조가 내재적 인과와 외재적 인과가 어디서부터 어떻게 무엇이
   어디까지 같고 다른지를 감각적 구조계층을 통해 포개어보는 비교대조원리라는거지.
   그래서 눈을 가장 첫째 감각으로 잡은거야."

Architecture:
  1. Uses `EmbodiedPygameSensoryWorld` + `LayeredSensoryInteractionSpine` as ELYSIA's
     Visual-Interactive Eye (`L_micro`, `L_meso`, `L_macro` + emergent boundary layer).
  2. Superposes internal `LinguisticCausalStructure` trajectories onto each observed
     entity's sensory column (`superpose_and_contrast`), separately evaluating:
       - `sync_micro`: surface appearance / color / local flicker match
       - `sync_meso` : kinetic flux / structural resistance / permeability / tactile reality match
       - `sync_macro`: persistent consequence & emergent phase-transition boundary match
  3. Explicitly identifies WHERE agreement ends and divergence begins (`matched_layers`
     vs `diverged_layers`), distinguishing illusions (`L_micro` matches, `L_meso/macro` fail)
     from cross-appearance physical invariants (`L_micro` differs, `L_meso/macro` match).
  4. Grounded full-conviction matches crystallize into `TrinitarianScaleRotorProtocol`
     (`s=2` Vision ⊗ Linguistic super-molecules inside `CausalField`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.sensory.layered_sensory_interaction_spine import (
    EmbodiedPygameSensoryWorld,
    LayeredSensoryInteractionSpine,
)
from core.sensory.trinitarian_scale_rotor_protocol import (
    TrinitarianScaleRotorProtocol,
)


DEFAULT_CONTRAST_DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_visual_linguistic_contrast_dashboard.png"
)


# ======================================================================
# 1. Linguistic Causal Structure & Superposition Verdict Contracts
# ======================================================================
@dataclass
class LinguisticCausalStructure:
    """Internal expectations encoded across all 3 sensory scales by a statement."""

    statement_id: str
    text_ko: str
    expected_micro_rgb: Tuple[float, float, float]
    expected_meso_triad: Tuple[float, float, float]  # (Red_Flux, Blue_Resist, Yellow_Perm)
    expected_beam_mm: float
    expected_macro_persist: float
    expected_boundary_shift: float


@dataclass
class SuperpositionContrastResult:
    """Layer-by-layer contrast showing up to where and from where internal/external match."""

    entity_id: str
    statement_id: str
    text_ko: str
    sync_micro: float
    sync_meso: float
    sync_macro: float
    overall_conviction: float
    verdict: str  # "CONVICTION_FULL_SYNC" | "DIVERGED_AT_MESO_MACRO" | "DIVERGED_AT_MICRO_ONLY" | "FULL_MISMATCH"
    matched_layers: Tuple[str, ...]
    diverged_layers: Tuple[str, ...]
    explanation_ko: str
    grounded_super_molecule_id: Optional[str] = None


def build_korean_statement_catalog() -> Dict[str, LinguisticCausalStructure]:
    """Structural expectations for Korean perceptual-causal propositions."""
    red_rgb = (0.88, 0.27, 0.27)
    green_rgb = (0.25, 0.84, 0.50)
    return {
        "stmt_red_wall": LinguisticCausalStructure(
            statement_id="stmt_red_wall",
            text_ko="빨간 벽이 길을 막는다",
            expected_micro_rgb=red_rgb,
            expected_meso_triad=(0.00, 1.00, 0.00),
            expected_beam_mm=0.90,
            expected_macro_persist=0.95,
            expected_boundary_shift=0.0,
        ),
        "stmt_red_crate": LinguisticCausalStructure(
            statement_id="stmt_red_crate",
            text_ko="빨간 상자가 밀려난다",
            expected_micro_rgb=red_rgb,
            expected_meso_triad=(0.85, 0.15, 0.00),
            expected_beam_mm=0.88,
            expected_macro_persist=0.92,
            expected_boundary_shift=0.0,
        ),
        "stmt_red_illusion": LinguisticCausalStructure(
            statement_id="stmt_red_illusion",
            text_ko="빨간 빛(허상)이 그대로 통과된다",
            expected_micro_rgb=red_rgb,
            expected_meso_triad=(0.00, 0.00, 0.93),
            expected_beam_mm=0.04,
            expected_macro_persist=0.05,
            expected_boundary_shift=0.0,
        ),
        "stmt_red_gate": LinguisticCausalStructure(
            statement_id="stmt_red_gate",
            text_ko="빨간 문이 버티다 갈라져 열린다",
            expected_micro_rgb=red_rgb,
            expected_meso_triad=(0.00, 0.70, 0.95),
            expected_beam_mm=0.26,
            expected_macro_persist=0.58,
            expected_boundary_shift=1.0,
        ),
        "stmt_green_wall": LinguisticCausalStructure(
            statement_id="stmt_green_wall",
            text_ko="초록 벽이 길을 막는다",
            expected_micro_rgb=green_rgb,
            expected_meso_triad=(0.00, 0.97, 0.01),
            expected_beam_mm=0.86,
            expected_macro_persist=0.93,
            expected_boundary_shift=0.0,
        ),
        "stmt_green_illusion": LinguisticCausalStructure(
            statement_id="stmt_green_illusion",
            text_ko="초록 빛(허상)이 그대로 통과된다",
            expected_micro_rgb=green_rgb,
            expected_meso_triad=(0.01, 0.01, 0.95),
            expected_beam_mm=0.05,
            expected_macro_persist=0.06,
            expected_boundary_shift=0.0,
        ),
    }


# ======================================================================
# 2. Visual-Linguistic Superposition & Contrast Engine
# ======================================================================
class VisualLinguisticSuperpositionEngine:
    """
    Superposes internal linguistic expectations (`LinguisticCausalStructure`) onto
    grounded visual-interactive sensory columns (`LayeredSensoryInteractionSpine`),
    identifying layer-specific synchronization & divergence before Trinitarian binding.
    """

    def __init__(self) -> None:
        self.sensory_spine = LayeredSensoryInteractionSpine()
        self.trinitarian = TrinitarianScaleRotorProtocol()
        self.pygame_snapshot: Optional[np.ndarray] = None
        self.inferred_affordances: Dict[str, str] = {}

    def observe_visual_world(self, headless: bool = True) -> np.ndarray:
        """Uses the visual-embodied eye (`EmbodiedPygameSensoryWorld`) as first mirror."""
        world = EmbodiedPygameSensoryWorld(headless=headless)
        snapshot, inferred = world.run_embodied_exploration(
            self.sensory_spine, max_steps_per_entity=16
        )
        self.pygame_snapshot = snapshot
        self.inferred_affordances = inferred
        return snapshot

    @staticmethod
    def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))

    @staticmethod
    def _deterministic_axis(tag: str) -> Tuple[Tuple[float, float, float], float]:
        d = hashlib.sha256(tag.encode("utf-8")).digest()
        v = np.array([float(b) / 255.0 - 0.5 for b in d[:3]], dtype=np.float64)
        n = float(np.linalg.norm(v))
        v = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else v / n
        return (float(v[0]), float(v[1]), float(v[2])), 0.6 + 1.0 * (float(d[3]) / 255.0)

    def superpose_and_contrast(
        self,
        entity_id: str,
        stmt: LinguisticCausalStructure,
        bind_on_conviction: bool = True,
    ) -> SuperpositionContrastResult:
        """
        Overlays `stmt` across L_micro / L_meso / L_macro of `entity_id` inside
        `self.sensory_spine.columns`, computing where they synchronize and where they diverge.
        """
        col = self.sensory_spine.columns[entity_id]

        # 1) L_micro comparison (surface RGB appearance)
        vis_micro = np.asarray(col.pre_interaction_vec[:3], dtype=np.float64)
        exp_micro = np.asarray(stmt.expected_micro_rgb, dtype=np.float64)
        micro_cos = self._cos_sim(vis_micro, exp_micro)
        micro_l1 = float(np.mean(np.abs(vis_micro - exp_micro)))
        sync_micro = float(np.clip(micro_cos - 0.45 * micro_l1, 0.0, 1.0))

        # 2) L_meso comparison (contact flux, resistance, permeability, visual-tactile beam)
        vis_meso = np.array(
            [
                float(col.chromatic_triad[0]),
                float(col.chromatic_triad[1]),
                float(col.chromatic_triad[2]),
                float(col.beam_micro_meso),
            ],
            dtype=np.float64,
        )
        exp_meso = np.array(
            [
                float(stmt.expected_meso_triad[0]),
                float(stmt.expected_meso_triad[1]),
                float(stmt.expected_meso_triad[2]),
                float(stmt.expected_beam_mm),
            ],
            dtype=np.float64,
        )
        meso_cos = self._cos_sim(vis_meso, exp_meso)
        meso_l1 = float(np.mean(np.abs(vis_meso - exp_meso)))
        sync_meso = float(np.clip(0.65 * meso_cos + 0.35 * (1.0 - meso_l1), 0.0, 1.0))

        # 3) L_macro comparison (persistent structural beam & emergent boundary transition)
        vis_b_flag = 1.0 if col.boundary_voxel_id is not None else 0.0
        sync_macro = float(
            np.clip(
                1.0
                - 0.70 * abs(float(col.beam_meso_macro) - float(stmt.expected_macro_persist))
                - 0.75 * abs(vis_b_flag - float(stmt.expected_boundary_shift)),
                0.0,
                1.0,
            )
        )

        # Weighted holistic conviction (physical causality weighs deeper than pure surface tint)
        overall_conviction = float(0.25 * sync_micro + 0.45 * sync_meso + 0.30 * sync_macro)

        matched: List[str] = []
        diverged: List[str] = []
        for lname, s_val, thr in (
            ("L_micro", sync_micro, 0.75),
            ("L_meso", sync_meso, 0.72),
            ("L_macro", sync_macro, 0.72),
        ):
            if s_val >= thr:
                matched.append(lname)
            else:
                diverged.append(lname)

        micro_ok = "L_micro" in matched
        meso_ok = "L_meso" in matched
        macro_ok = "L_macro" in matched

        if micro_ok and meso_ok and macro_ok:
            verdict = "CONVICTION_FULL_SYNC"
            exp_ko = (
                f"'{stmt.text_ko}'가 눈으로 본 표면(L_micro={sync_micro:.2f}), "
                f"신체 상호작용(L_meso={sync_meso:.2f}), 지속 변화(L_macro={sync_macro:.2f})와 "
                "전 계층에서 일치하여 납득됨."
            )
        elif micro_ok and not (meso_ok and macro_ok):
            verdict = "DIVERGED_AT_MESO_MACRO"
            exp_ko = (
                f"'{stmt.text_ko}': 표면 색·모양(L_micro={sync_micro:.2f})까지는 같지만, "
                f"부딪침·통과(L_meso={sync_meso:.2f}) 및 지속성(L_macro={sync_macro:.2f})에서 어긋남."
            )
        elif (not micro_ok) and meso_ok and macro_ok:
            verdict = "DIVERGED_AT_MICRO_ONLY"
            exp_ko = (
                f"'{stmt.text_ko}': 표면 색(L_micro={sync_micro:.2f})은 다르지만, "
                f"실제 작동 원리(L_meso={sync_meso:.2f}, L_macro={sync_macro:.2f})는 일치함."
            )
        else:
            verdict = "FULL_MISMATCH"
            exp_ko = (
                f"'{stmt.text_ko}': 표면(L_micro={sync_micro:.2f})과 인과 구조"
                f"(L_meso={sync_meso:.2f}, L_macro={sync_macro:.2f}) 모두 일치하지 않음."
            )

        grounded_mol_id: Optional[str] = None
        if verdict == "CONVICTION_FULL_SYNC" and bind_on_conviction:
            vis_id = f"vis_ctx::{entity_id}"
            ling_id = f"ling_ctx::{stmt.statement_id}"
            if vis_id not in self.trinitarian.nodes:
                ax_v, ang_v = self._deterministic_axis(vis_id)
                self.trinitarian.register_atom(
                    vis_id,
                    ax_v,
                    ang_v,
                    tuple(float(x) for x in col.chromatic_triad),
                )
            if ling_id not in self.trinitarian.nodes:
                ax_l, ang_l = self._deterministic_axis(ling_id)
                self.trinitarian.register_atom(
                    ling_id, ax_l, ang_l, stmt.expected_meso_triad
                )
            mol = self.trinitarian.resonate_and_bind(
                vis_id,
                ling_id,
                bond_coherence=overall_conviction,
                molecule_id=f"convinced::{entity_id}__{stmt.statement_id}",
            )
            grounded_mol_id = mol.node_id

        return SuperpositionContrastResult(
            entity_id=entity_id,
            statement_id=stmt.statement_id,
            text_ko=stmt.text_ko,
            sync_micro=sync_micro,
            sync_meso=sync_meso,
            sync_macro=sync_macro,
            overall_conviction=overall_conviction,
            verdict=verdict,
            matched_layers=tuple(matched),
            diverged_layers=tuple(diverged),
            explanation_ko=exp_ko,
            grounded_super_molecule_id=grounded_mol_id,
        )

    def ground_entity_to_best_statement(
        self,
        entity_id: str,
        catalog: Dict[str, LinguisticCausalStructure],
    ) -> SuperpositionContrastResult:
        """Finds which statement genuinely convinces all sensory layers for `entity_id`."""
        trials = [
            self.superpose_and_contrast(entity_id, st, bind_on_conviction=False)
            for st in catalog.values()
        ]
        best = max(trials, key=lambda r: r.overall_conviction)
        return self.superpose_and_contrast(
            entity_id, catalog[best.statement_id], bind_on_conviction=True
        )


# ======================================================================
# 3. 4-Panel Dashboard Renderer (Clean ASCII/English Chart Typography)
# ======================================================================
def render_visual_linguistic_contrast_dashboard(
    engine: VisualLinguisticSuperpositionEngine,
    key_contrasts: List[SuperpositionContrastResult],
    conviction_matrix: np.ndarray,
    entity_ids: List[str],
    stmt_ids: List[str],
    output_path: Path = DEFAULT_CONTRAST_DASHBOARD_PATH,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.4), dpi=140)
    fig.patch.set_facecolor("#0d1117")
    fig.subplots_adjust(hspace=0.35, wspace=0.22, top=0.92, bottom=0.08, left=0.07, right=0.97)

    # Panel 1: Layer-by-Layer Synchronization Comparison
    ax1 = axes[0, 0]
    ax1.set_facecolor("#161b22")
    cx = np.arange(len(key_contrasts))
    w3 = 0.25
    s_mic = [r.sync_micro for r in key_contrasts]
    s_mes = [r.sync_meso for r in key_contrasts]
    s_mac = [r.sync_macro for r in key_contrasts]
    ax1.bar(cx - w3, s_mic, width=w3, color="#ff7b72", label="L_micro (Surface RGB)")
    ax1.bar(cx, s_mes, width=w3, color="#58a6ff", label="L_meso (Tactile/Flux Physics)")
    ax1.bar(cx + w3, s_mac, width=w3, color="#7ee787", label="L_macro (Persistence/Boundary)")
    short_case_labels = [
        f"{r.entity_id}\nvs {r.statement_id.replace('stmt_', '')}" for r in key_contrasts
    ]
    ax1.set_ylim(0.0, 1.32)
    ax1.set_xticks(cx)
    ax1.set_xticklabels(short_case_labels, color="#e6edf3", fontsize=7.8)
    ax1.set_ylabel("Layer Synchronization (0..1)", color="#c9d1d9", fontsize=9)
    ax1.set_title("Where Vision & Language Synchronize vs Diverge", color="#e6edf3", fontsize=10.5)
    ax1.tick_params(colors="#c9d1d9", labelsize=8)
    ax1.legend(loc="upper center", ncol=3, facecolor="#21262d", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=7.0)

    # Panel 2: Entity x Statement Multi-Scale Conviction Matrix
    ax2 = axes[0, 1]
    ax2.set_facecolor("#161b22")
    im = ax2.imshow(conviction_matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    for r in range(len(entity_ids)):
        for c in range(len(stmt_ids)):
            val = float(conviction_matrix[r, c])
            txt_col = "#0d1117" if val > 0.68 else "#e6edf3"
            ax2.text(c, r, f"{val:.2f}", ha="center", va="center", color=txt_col, fontsize=8.0, fontweight="bold")
    ax2.set_xticks(np.arange(len(stmt_ids)))
    ax2.set_xticklabels([s.replace("stmt_", "") for s in stmt_ids], color="#e6edf3", fontsize=7.6, rotation=18)
    ax2.set_yticks(np.arange(len(entity_ids)))
    ax2.set_yticklabels(entity_ids, color="#e6edf3", fontsize=8.2)
    ax2.set_title("Multi-Scale Conviction Heatmap (Exact Grounding Diagonal)", color="#e6edf3", fontsize=10.5)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#c9d1d9", labelsize=7.5)

    # Panel 3: Rendered Pygame Eye Arena Snapshot
    ax3 = axes[1, 0]
    ax3.set_facecolor("#161b22")
    if engine.pygame_snapshot is not None:
        ax3.imshow(engine.pygame_snapshot)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title("Eye as First Mirror: Observed Visual-Interactive World", color="#e6edf3", fontsize=10.5)

    # Panel 4: Layer-Locus Verdict Breakdown Table
    ax4 = axes[1, 1]
    ax4.set_facecolor("#161b22")
    ax4.axis("off")
    ax4.set_title("Layer-Contrast Verdicts & Trinitarian Grounding", color="#e6edf3", fontsize=10.5)
    y_pos = 0.90
    for r in key_contrasts:
        v_col = "#7ee787" if r.verdict == "CONVICTION_FULL_SYNC" else (
            "#ffa657" if r.verdict == "DIVERGED_AT_MICRO_ONLY" else "#ff7b72"
        )
        m_str = ",".join(r.matched_layers) if r.matched_layers else "none"
        d_str = ",".join(r.diverged_layers) if r.diverged_layers else "none"
        ax4.text(
            0.03,
            y_pos,
            f"[{r.entity_id} vs {r.statement_id}] -> {r.verdict}",
            color=v_col,
            fontsize=8.4,
            fontweight="bold",
            transform=ax4.transAxes,
        )
        y_pos -= 0.065
        ax4.text(
            0.06,
            y_pos,
            f"Sync=(micro={r.sync_micro:.2f}, meso={r.sync_meso:.2f}, macro={r.sync_macro:.2f}) | "
            f"Match=[{m_str}] Diverge=[{d_str}]",
            color="#c9d1d9",
            fontsize=7.7,
            family="monospace",
            transform=ax4.transAxes,
        )
        y_pos -= 0.055
        if r.grounded_super_molecule_id is not None:
            ax4.text(
                0.06,
                y_pos,
                f"Grounded Rotor: {r.grounded_super_molecule_id}",
                color="#58a6ff",
                fontsize=7.6,
                transform=ax4.transAxes,
            )
        else:
            ax4.text(
                0.06,
                y_pos,
                "Binding Refused (Partial / Mismatched Conviction)",
                color="#8b949e",
                fontsize=7.6,
                transform=ax4.transAxes,
            )
        y_pos -= 0.085

    ax4.text(
        0.03,
        0.03,
        f"Trinitarian Field: {len(engine.trinitarian.nodes)} Nodes | {len(engine.trinitarian.causal_field.beams)} Beams",
        color="#d2a8ff",
        fontsize=8.3,
        fontweight="bold",
        transform=ax4.transAxes,
    )

    fig.savefig(output_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path
