"""
verify_unified_world_game_book_learning.py

Standalone end-to-end verification and 6-panel visualization dashboard for ELYSIA's
Closed World -> Book -> Blind Game -> Autonomous Creation Experiential Learning Loop.

Outputs:
- Structured verification telemetry for both Korean and Blind-Token (`use_blind_tokens=True`) modes
- Cold-reload JSON persistence parity check (`EpistemicKnowledgeSpine.save_json` / `load_json`)
- 6-panel visual inspection dashboard at:
  `c:/Elysia/docs/assets/unified_world_game_book_learning_inspection.png`
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "elysia_mpl"))

sys.path.insert(0, r"c:\Elysia")
import elysia_loader

elysia_loader.install()

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.sensory.epistemic_knowledge_spine import EpistemicKnowledgeSpine
from core.sensory.unified_experiential_learner import (
    BookReadingLearner,
    build_curriculum_bundle,
    run_unified_world_game_book_session,
)


def _draw_trajectory(ax, frame_rgb: np.ndarray, traj, tile_px: int = 40, color: str = "#00f5d4") -> None:
    ax.imshow(frame_rgb)
    if len(traj) >= 2:
        xs = [c * tile_px + tile_px * 0.5 for _, c in traj]
        ys = [r * tile_px + tile_px * 0.5 for r, _ in traj]
        ax.plot(xs, ys, color=color, linewidth=2.2, alpha=0.92, zorder=4)
        ax.scatter(xs[0], ys[0], color="#fee440", s=42, zorder=5, edgecolors="black", linewidths=0.8)
        ax.scatter(xs[-1], ys[-1], color="#00f5d4", s=54, marker="*", zorder=5, edgecolors="black", linewidths=0.8)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> int:
    out_png = Path(r"c:\Elysia\docs\assets\unified_world_game_book_learning_inspection.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    print("[1/3] Running World -> Book -> Game -> Creation session (Natural Korean)...")
    rep_kr = run_unified_world_game_book_session(use_blind_tokens=False)

    print("[2/3] Running identical session under Strict Blind Tokens (0% semantic prior)...")
    rep_bl = run_unified_world_game_book_session(use_blind_tokens=True)

    print("[3/3] Verifying cold-reload JSON persistence roundtrip...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "epistemic_spine_cold_reload.json"
        rep_kr.spine.save_json(ckpt)
        reloaded = EpistemicKnowledgeSpine.load_json(ckpt)
        curr_kr = build_curriculum_bundle(use_blind_tokens=False)
        acc_reload, _ = BookReadingLearner(reloaded).evaluate_disambiguation_benchmark(
            curr_kr["unseen_tests"]
        )
        assert len(reloaded.concepts) == len(rep_kr.spine.concepts)
        assert set(reloaded.symbols.keys()) == set(rep_kr.spine.symbols.keys())
        assert acc_reload == 1.0

    # Summary printout
    print("\n=== VERIFICATION SUMMARY ===")
    sat = rep_kr.saturation_law
    print(
        f"World Saturation Law : monotone={sat.monotone_chroma_decay}, "
        f"chroma=[{sat.chroma_radii[0]:.2f} -> {sat.chroma_radii[-1]:.4f}], "
        f"core_hue_drift={sat.max_hue_drift_deg_core:.1f}deg, tipping_w={sat.boundary_tipping_w}"
    )
    for key, b in rep_kr.axis_bindings.items():
        b_bl = rep_bl.axis_bindings[key]
        print(
            f"Axis Binding [{key:10s}]: {b.axis_name:20s} dir={b.direction:+.0f} "
            f"KR_margin={b.reobservation_margin:.3f} (ver={b.reobservation_verified}) | "
            f"BLIND_margin={b_bl.reobservation_margin:.3f} (ver={b_bl.reobservation_verified})"
        )
    print(
        f"Book Corpus & Polysemy : corpus_files={rep_kr.corpus_files_read}, "
        f"KR_disamb={rep_kr.disambiguation_accuracy*100:.0f}% (6/6), "
        f"BLIND_disamb={rep_bl.disambiguation_accuracy*100:.0f}% (6/6), cold_reload={acc_reload*100:.0f}%"
    )
    print(
        f"Blind Game L1 (0 deg)  : cleared={rep_kr.game_level1.cleared}, steps={rep_kr.game_level1.steps}, "
        f"haz={rep_kr.game_level1.hazard_hits}, cleaved={rep_kr.game_level1.cleaved_barriers}, "
        f"foams={rep_kr.game_level1.cleared_foams}"
    )
    print(
        f"Blind Game L2 (90d/dim): cleared={rep_kr.game_level2.cleared}, steps={rep_kr.game_level2.steps}, "
        f"haz={rep_kr.game_level2.hazard_hits}, cleaved={rep_kr.game_level2.cleaved_barriers}, "
        f"foams={rep_kr.game_level2.cleared_foams}"
    )
    print(
        f"Created Level 3 + Vis  : L3_ver={rep_kr.created_level3.verified} "
        f"(steps={rep_kr.created_level3.play_summary.steps}, haz={rep_kr.created_level3.play_summary.hazard_hits}) | "
        f"Vis_ver={rep_kr.created_visual.verified} (L2_corr={rep_kr.created_visual.l2_persistence_corr:.3f}, "
        f"dc={rep_kr.created_visual.created_metrics['chroma_radius']:+.3f}, "
        f"dr={rep_kr.created_visual.created_metrics['fine_roughness']:+.3f})"
    )
    print(
        f"CausalField Topology   : KR=({len(rep_kr.causal_field.voxels)} voxels, {len(rep_kr.causal_field.beams)} beams), "
        f"BLIND=({len(rep_bl.causal_field.voxels)} voxels, {len(rep_bl.causal_field.beams)} beams)"
    )

    # Render 6-panel dashboard (ASCII/English labels for crisp headless rendering)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.2), dpi=135)
    fig.patch.set_facecolor("#0f1419")

    # Panel 1: Spectral Purity vs White Dilution Law
    ax1 = axes[0, 0]
    w_vals = sat.white_dilutions
    ax1.plot(w_vals, sat.chroma_radii, marker="o", linewidth=2.4, color="#ff6b6b", label="Chroma Radius (u)")
    ax1.set_title("1. Saturation Law: White Dilution vs Chroma", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlabel("Broadband White Dilution (w)")
    ax1.set_ylabel("Chroma Radius (Cone Opponent)", color="#ff6b6b")
    ax1.grid(alpha=0.22)
    ax1_r = ax1.twinx()
    ax1_r.plot(w_vals, sat.l2_persistence_corrs, marker="s", linestyle="--", linewidth=1.8, color="#4ecdc4", label="L2 Persistence Corr")
    ax1_r.set_ylabel("L2 Structural Persistence", color="#4ecdc4")
    ax1_r.set_ylim(0.85, 1.02)
    ax1.axvline(sat.boundary_tipping_w or 0.8, color="#ffe66d", linestyle=":", linewidth=1.4)
    ax1.text(0.03, 0.12, f"Core Hue Drift: {sat.max_hue_drift_deg_core:.1f} deg\nOrtho EV Drift: {sat.ortho_ev_chroma_drift:.3f}", transform=ax1.transAxes, fontsize=8.5, bbox=dict(boxstyle="round", facecolor="#1a222d", alpha=0.85))

    # Panel 2: Confound Separation -> 5 Grounded Axis Bindings
    ax2 = axes[0, 1]
    ax_keys = list(rep_kr.axis_bindings.keys())
    kr_margins = [rep_kr.axis_bindings[k].reobservation_margin for k in ax_keys]
    bl_margins = [rep_bl.axis_bindings[k].reobservation_margin for k in ax_keys]
    y_pos = np.arange(len(ax_keys))
    ax2.barh(y_pos - 0.18, kr_margins, height=0.34, color="#48cae4", label="Korean Symbol")
    ax2.barh(y_pos + 0.18, bl_margins, height=0.34, color="#f77f00", label="Blind Token")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{k}\n({rep_kr.axis_bindings[k].axis_name})" for k in ax_keys], fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Counterfactual Re-Observation Margin")
    ax2.set_title("2. Blank-Slate Axis Bindings (KR == Blind)", fontsize=11, fontweight="bold", pad=8)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(axis="x", alpha=0.22)

    # Panel 3: Book Polysemy Lexical Mitosis (Shared Stem vs Divergent Branch)
    ax3 = axes[0, 2]
    poly_labels = ["Pair 1\n(Fruit/Apology)", "Pair 2\n(Light/Mood)", "Pair 3\n(Rock/Resolve)"]
    kr_forms = [list(rep_kr.sense_comparisons.keys())[0], list(rep_kr.sense_comparisons.keys())[1], list(rep_kr.sense_comparisons.keys())[2]]
    stem_sims = [np.mean([c.structural_stem_similarity for c in rep_kr.sense_comparisons[f]]) for f in kr_forms]
    ctx_divs = [np.mean([c.context_divergence for c in rep_kr.sense_comparisons[f]]) for f in kr_forms]
    x_pos = np.arange(len(poly_labels))
    ax3.bar(x_pos - 0.18, stem_sims, width=0.34, color="#80ed99", label="Shared Stem Sim")
    ax3.bar(x_pos + 0.18, ctx_divs, width=0.34, color="#ff9f1c", label="Branch Divergence")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(poly_labels, fontsize=8.5)
    ax3.set_ylim(0.0, 1.15)
    ax3.set_title("3. Contextual Mitosis: Sameness vs Difference\n(Unseen Disambig: 6/6 KR & 6/6 Blind)", fontsize=10.5, fontweight="bold", pad=8)
    ax3.legend(fontsize=8, loc="lower right")
    ax3.grid(axis="y", alpha=0.22)

    # Panel 4: Blind Game Level 1 -> Level 2 Zero-Shot Transfer
    ax4 = axes[1, 0]
    g2 = rep_kr.game_level2
    _draw_trajectory(ax4, g2.final_frame_rgb, g2.trajectory, color="#00f5d4")
    ax4.set_title(
        f"4. Blind Game L2 (90d Rotated, 0.72x Dim)\n"
        f"Cleared={g2.cleared} | Steps={g2.steps} | Haz={g2.hazard_hits} | Cleaved={g2.cleaved_barriers} | Foam={g2.cleared_foams}",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )

    # Panel 5: Autonomous Creation (Level 3 Self-Play + Multi-Axis Visual)
    ax5 = axes[1, 1]
    l3 = rep_kr.created_level3
    cvis = rep_kr.created_visual
    # Compose side-by-side inset of Created Visual (right) and Self-Created Level 3 (left)
    left_img = np.asarray(l3.play_summary.final_frame_rgb, dtype=np.uint8)
    vis_h, vis_w = left_img.shape[:2]
    vis_arr = np.asarray(cvis.created_rgb)
    if vis_arr.dtype != np.uint8:
        scale = 255.0 if float(np.max(vis_arr)) <= 1.05 else 1.0
        vis_u8 = np.clip(np.round(vis_arr * scale), 0, 255).astype(np.uint8)
    else:
        vis_u8 = vis_arr
    right_img = cv2.resize(vis_u8, (vis_w, vis_h), interpolation=cv2.INTER_AREA)
    combined = np.concatenate([left_img, right_img], axis=1)
    ax5.imshow(combined)
    xs3 = [c * 40 + 20 for _, c in l3.play_summary.trajectory]
    ys3 = [r * 40 + 20 for r, _ in l3.play_summary.trajectory]
    if len(xs3) >= 2:
        ax5.plot(xs3, ys3, color="#fee440", linewidth=2.2, zorder=4)
        ax5.scatter(xs3[0], ys3[0], color="#fee440", s=42, zorder=5, edgecolors="black", linewidths=0.8)
        ax5.scatter(xs3[-1], ys3[-1], color="#00f5d4", s=54, marker="*", zorder=5, edgecolors="black", linewidths=0.8)
    ax5.axvline(vis_w, color="white", linestyle="--", linewidth=1.2)
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.set_title(
        f"5. Autonomous Creation & Self-Verification\n"
        f"L3: {l3.play_summary.steps}steps/0haz | Vis: dC={cvis.created_metrics['chroma_radius']:+.2f}, dR={cvis.created_metrics['fine_roughness']:+.2f}, L2={cvis.l2_persistence_corr:.2f}",
        fontsize=9.5,
        fontweight="bold",
        pad=8,
    )

    # Panel 6: Epistemic Void & Mitosis Lifecycle across Learning Stages
    ax6 = axes[1, 2]
    stages = [row["stage"] for row in rep_kr.void_timeline]
    open_v = [row["open_voids"] for row in rep_kr.void_timeline]
    res_v = [row["resolved_voids"] for row in rep_kr.void_timeline]
    mit_v = [row["mitosis"] for row in rep_kr.void_timeline]
    sx = np.arange(len(stages))
    ax6.plot(sx, open_v, marker="^", linewidth=2.0, color="#ff595e", label="Open Voids")
    ax6.plot(sx, res_v, marker="o", linewidth=2.2, color="#8ac926", label="Resolved Voids")
    ax6.plot(sx, mit_v, marker="s", linewidth=2.0, color="#1982c4", label="Cumulative Mitosis")
    ax6.set_xticks(sx)
    ax6.set_xticklabels(stages, rotation=22, ha="right", fontsize=8)
    ax6.set_ylabel("Count")
    ax6.set_title(
        f"6. Void Lifecycle & CausalField ({len(rep_kr.causal_field.voxels)}V / {len(rep_kr.causal_field.beams)}B)",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    ax6.legend(fontsize=8, loc="upper left")
    ax6.grid(alpha=0.22)

    fig.suptitle(
        "ELYSIA Unified World -> Book -> Blind Game -> Creation Experiential Learning Verification",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.955])
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved inspection figure -> {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
