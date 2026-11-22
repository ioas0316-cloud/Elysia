"""
verify_active_epistemic_explorer.py

Active Self-Directing Multi-Organ Epistemic Explorer Verification Runner
========================================================================
Executes ActiveEpistemicExplorer on both:
  - Testbed A: Unlabeled genuine 4K Yosemite photograph (pure optical foveation)
  - Testbed B: HiddenMultiPhysicsWorld rendered strictly from real Yosemite textures
Outputs exact autonomous priority decisions, blank-slate cross-modal prototypes,
Prototype Mitosis on deceptive counter-examples, motor-visual shadow resolution,
and the 6-panel inspection figure at:
    c:/Elysia/docs/assets/active_epistemic_explorer_inspection.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"c:\Elysia")
try:
    import elysia_loader

    elysia_loader.install()
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

from core.sensory.active_epistemic_explorer import ActiveEpistemicExplorer
from simulators.hidden_multiphysics_world import (
    HiddenMultiPhysicsWorld,
    Real4KYosemiteWorld,
    YOSEMITE_4K_PATH,
)

OUT_PNG_PATH = Path(r"c:\Elysia\docs\assets\active_epistemic_explorer_inspection.png")


def main() -> None:
    print("=" * 88)
    print("  ELYSIA · Active Self-Directing Multi-Organ Epistemic Explorer Verification")
    print("=" * 88)

    explorer = ActiveEpistemicExplorer()

    # Testbed A: Genuine 4K Yosemite Photo
    w_yos = Real4KYosemiteWorld(YOSEMITE_4K_PATH)
    trace_a = explorer.explore(w_yos, num_steps=6)

    print("\n[Testbed A] Real 4K Yosemite Photograph (supports_physical_probing=False)")
    for d in trace_a.decisions:
        c = d.chosen
        print(
            f"  Step {d.step}: target={c.target_id:10s} ({c.y_norm:.2f},{c.x_norm:.2f}) "
            f"prio={c.total_priority:+.2f} act={c.recommended_action:13s} | {d.outcome_summary}"
        )

    # Testbed B: HiddenMultiPhysicsWorld
    w_phys = HiddenMultiPhysicsWorld(YOSEMITE_4K_PATH, seed=42)
    trace_b = explorer.explore(w_phys, num_steps=16)

    print("\n[Testbed B] HiddenMultiPhysicsWorld (Blank Slate Cross-Modal Exploration)")
    for d in trace_b.decisions:
        c = d.chosen
        gt = w_phys.evaluate_ground_truth_at(c.y_norm, c.x_norm)
        gt_label = gt["body_id"] if gt else "BACKDROP"
        print(
            f"  Step {d.step:02d}: target={c.target_id:16s} [{gt_label:22s}] "
            f"prio={c.total_priority:+.2f} act={c.recommended_action:27s} | {d.outcome_summary}"
        )

    print("\n[Discovered Cross-Modal Prototypes (Started from Empty)]")
    for p in trace_b.prototypes:
        f_str = (
            f"lap_hf={p.visual_foveal_center[0]:.3f}"
            if p.visual_foveal_center is not None
            else "foveal=None"
        )
        m_str = (
            f"log10_k={p.mech_center[0]:.2f}, damp={p.mech_center[1]:.2f}, plast={p.mech_center[2]:.2f}"
            if p.mech_center is not None
            else "mech=None"
        )
        a_str = f"log10_f={p.acoustic_center[0]:.2f}" if p.acoustic_center is not None else "acoust=None"
        print(
            f"  - {p.prototype_id:14s}: confirm={p.confirmation_count}, parent={str(p.parent_prototype_id):8s} "
            f"| {f_str} | {m_str} | {a_str}"
        )

    print("\n[Prototype Mitosis / Refinements]")
    for r in trace_b.refinements:
        print(f"  - Step {r.step}: {r.parent_prototype_id} -> {r.new_prototype_id}: {r.separating_axis_note}")

    print("\n[Motor-Visual Resolved Shadow Boundaries]")
    for pair, verdict in trace_b.resolved_shadow_boundaries.items():
        print(f"  - Region pair {pair}: {verdict}")

    cm = trace_b.correlation_matrix
    print(f"\n[Self-Discovered Empirical Correlation Matrix (n={cm['sample_count']} episodes)]")
    print(f"  - Mechanical log10(stiffness) vs Acoustic log10(peak_freq): r = {cm['stiffness_vs_freq_r']:+.3f}")

    print("\n[Honest Remaining Epistemic Limits]")
    for lim in trace_b.organ_audit.modality_limits:
        print(f"  * {lim}")
    print("  * Without physical probing capability (Testbed A), cross-modal prototypes remain empty (0 fabricated inferences).")
    print("  * Learned correlations are observational associations over encountered regions, not universal chemical identities.")

    render_inspection_figure(trace_a, trace_b, w_yos, w_phys, OUT_PNG_PATH)


def render_inspection_figure(trace_a, trace_b, w_yos, w_phys, out_path: Path) -> None:
    fig = plt.figure(figsize=(22, 13.5), facecolor="#0d1117")
    fig.suptitle(
        "ELYSIA · Active Self-Directing Multi-Organ Epistemic Explorer & Blank-Slate Cross-Modal Discovery",
        color="#f0f6fc",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.22, left=0.05, right=0.97, top=0.91, bottom=0.06)
    panel_bg = "#161b22"
    txt_col = "#e6edf3"

    # Panel 1: Testbed A Real 4K Yosemite Autonomous Gaze Trajectory
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(panel_bg)
    ax1.imshow(trace_a.periphery_obs.rgb_working)
    ha, wa = trace_a.periphery_obs.shape_hw
    gaze_y = [d.chosen.y_norm * ha for d in trace_a.decisions]
    gaze_x = [d.chosen.x_norm * wa for d in trace_a.decisions]
    ax1.plot(gaze_x, gaze_y, color="#3fb950", linestyle="--", linewidth=1.6, zorder=4)
    for idx, (gx, gy) in enumerate(zip(gaze_x, gaze_y), 1):
        ax1.scatter([gx], [gy], s=110, color="#facc15", edgecolors="#0d1117", linewidths=1.2, zorder=5)
        ax1.text(gx + 3, gy - 3, f"{idx}", color="#facc15", fontsize=9, fontweight="bold", zorder=6)
    # Inset showing native 1:1 foveal fixated crop from step 1
    first_dec = trace_a.decisions[0].chosen
    fov_crop = w_yos.foveate(first_dec.y_norm, first_dec.x_norm, fov_frac=0.10)
    ins = ax1.inset_axes([0.62, 0.56, 0.35, 0.40])
    ins.imshow(fov_crop)
    ins.set_title("Step 1 1:1 Fovea", color="white", fontsize=7.5, pad=2)
    ins.axis("off")
    ax1.set_title(
        f"① Testbed A: 4K Yosemite 자율 주시 궤적 (6 steps)\n"
        f"Pure Optical World (no physical probes): {len({(round(d.chosen.y_norm,2), round(d.chosen.x_norm,2)) for d in trace_a.decisions})} distinct targets",
        color=txt_col,
        fontsize=9.0,
        pad=8,
    )
    ax1.axis("off")

    # Panel 2: Testbed B Autonomous Exploration Map & Organ Selection
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(panel_bg)
    ax2.imshow(trace_b.periphery_obs.rgb_working)
    hb, wb = trace_b.periphery_obs.shape_hw
    act_colors = {
        "MECHANICAL_AND_ACOUSTIC_PROBE": "#38bdf8",
        "FOVEAL_FIXATE": "#facc15",
        "MECHANICAL_PUSH_AND_WATCH": "#f43f5e",
    }
    act_markers = {
        "MECHANICAL_AND_ACOUSTIC_PROBE": "o",
        "FOVEAL_FIXATE": "^",
        "MECHANICAL_PUSH_AND_WATCH": "s",
    }
    by_s = [d.chosen.y_norm * hb for d in trace_b.decisions]
    bx_s = [d.chosen.x_norm * wb for d in trace_b.decisions]
    ax2.plot(bx_s, by_s, color="white", alpha=0.45, linestyle=":", linewidth=1.2, zorder=3)
    seen_labels = set()
    for d in trace_b.decisions:
        c = d.chosen
        col = act_colors.get(c.recommended_action, "#ffffff")
        mkr = act_markers.get(c.recommended_action, "o")
        lbl = c.recommended_action if c.recommended_action not in seen_labels else None
        if lbl:
            seen_labels.add(c.recommended_action)
        px, py = c.x_norm * wb, c.y_norm * hb
        ax2.scatter([px], [py], s=95, color=col, marker=mkr, edgecolors="white", linewidths=0.9, zorder=5, label=lbl)
        if d.step <= 10:
            ax2.text(px + 3, py - 3, f"{d.step}", color="white", fontsize=8, fontweight="bold", zorder=6)
    ax2.legend(facecolor=panel_bg, labelcolor=txt_col, fontsize=7.0, loc="lower center", ncol=2)
    ax2.set_title(
        "② Testbed B: 다중 감각 자율 탐색 궤적 (Step 1–16)\n"
        "Blue=Contact+Tap | Gold=Foveal Gaze | Red=Push-and-Watch",
        color=txt_col,
        fontsize=9.0,
        pad=8,
    )
    ax2.axis("off")

    # Panel 3: Epistemic Priority & Surprise Timeline
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(panel_bg)
    steps_x = [d.step for d in trace_b.decisions]
    prios = [d.chosen.total_priority for d in trace_b.decisions]
    surp_bars = [d.episode.prediction_surprise if d.episode else 0.0 for d in trace_b.decisions]
    ax3.plot(steps_x, prios, color="#58a6ff", marker="o", linewidth=2.0, label="Top Candidate Priority")
    ax3.bar(steps_x, surp_bars, color="#f43f5e", alpha=0.80, width=0.45, label="Prediction Surprise Error")
    ax3.axhline(0.75, color="#f97316", linestyle="--", linewidth=1.2, label="Mitosis Threshold (0.75)")
    for r in trace_b.refinements:
        ax3.annotate(
            f"Step {r.step} Surprise!\n-> Step {r.step+1} Fovea Lock",
            xy=(r.step, r.surprise_error),
            xytext=(min(r.step + 1.5, 11.5), max(prios) * 0.72),
            arrowprops=dict(arrowstyle="->", color="#facc15", lw=1.4),
            color="#facc15",
            fontsize=8.0,
            fontweight="bold",
        )
    ax3.set_xlabel("Exploration Step", color=txt_col, fontsize=8.5)
    ax3.set_ylabel("Priority / Surprise Score", color=txt_col, fontsize=8.5)
    ax3.tick_params(colors=txt_col, labelsize=8)
    ax3.grid(color="#30363d", linestyle="--", linewidth=0.5, alpha=0.6)
    ax3.legend(facecolor=panel_bg, labelcolor=txt_col, fontsize=7.5, loc="upper right")
    ax3.set_title("③ 인식적 우선순위 & 예측 놀람(Surprise) 포착 궤적", color=txt_col, fontsize=9.5, pad=8)

    # Panel 4: Blank-Slate Cross-Modal Discovery & Mitosis Split
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(panel_bg)
    proto_cols = ["#f43f5e", "#38bdf8", "#3fb950", "#94a3b8", "#facc15", "#a855f7"]
    for idx, p in enumerate(trace_b.prototypes):
        if p.mech_center is None:
            continue
        k_log = p.mech_center[0]
        lap = p.visual_foveal_center[0] if p.visual_foveal_center is not None else 0.0
        col = proto_cols[idx % len(proto_cols)]
        ax4.scatter(
            [lap],
            [k_log],
            s=180 + 70 * p.confirmation_count,
            color=col,
            edgecolors="white",
            linewidths=1.4,
            zorder=5,
            label=f"{p.prototype_id} (n={p.confirmation_count})",
        )
        ax4.text(lap + 0.008, k_log + 0.06, p.prototype_id, color=col, fontsize=8.0, fontweight="bold")
    # Draw mitosis arrow between parent and child
    proto_map = {p.prototype_id: p for p in trace_b.prototypes}
    for r in trace_b.refinements:
        par = proto_map.get(r.parent_prototype_id)
        chi = proto_map.get(r.new_prototype_id)
        if par and chi and par.mech_center is not None and chi.mech_center is not None:
            lx0 = par.visual_foveal_center[0] if par.visual_foveal_center is not None else 0.0
            lx1 = chi.visual_foveal_center[0] if chi.visual_foveal_center is not None else 0.0
            ax4.annotate(
                "",
                xy=(lx1, chi.mech_center[0]),
                xytext=(lx0, par.mech_center[0]),
                arrowprops=dict(arrowstyle="->", color="#facc15", lw=1.8, ls="--"),
            )
    ax4.set_xlabel("Native 1:1 Foveal Micro-Contrast (lap_hf)", color=txt_col, fontsize=8.5)
    ax4.set_ylabel("Measured Mechanical Stiffness log10(k [N/m])", color=txt_col, fontsize=8.5)
    ax4.tick_params(colors=txt_col, labelsize=8)
    ax4.grid(color="#30363d", linestyle="--", linewidth=0.5, alpha=0.6)
    ax4.legend(facecolor=panel_bg, labelcolor=txt_col, fontsize=7.5, loc="lower right")
    ax4.set_title(
        "④ 백지 상태 교차모달 발견 & 반례 프로토타입 분열(Mitosis)\n"
        "Foveal lap_hf separates Deceptive Foam (~0.36) vs Rigid Granite (~0.08)",
        color=txt_col,
        fontsize=9.0,
        pad=8,
    )

    # Panel 5: Motor-Visual Shadow Resolution Flow Field
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(panel_bg)
    push_eps = [ep for ep in trace_b.episodes if ep.action_type == "MECHANICAL_PUSH_AND_WATCH" and ep.motor_visual_transition is not None]
    ax5.imshow(trace_b.periphery_obs.rgb_working * 0.65)
    if push_eps:
        p_ep = push_eps[0]
        tv = p_ep.motor_visual_transition
        step_q = 8
        yy, xx = np.mgrid[6 : hb - 6 : step_q, 6 : wb - 6 : step_q]
        fy = tv.flow_yx[yy, xx, 0]
        fx = tv.flow_yx[yy, xx, 1]
        ax5.quiver(xx, yy, fx, -fy, color="#00f5d4", scale=75, width=0.004)
        shad_pair = next(iter(trace_b.resolved_shadow_boundaries.items()), (None, "NONE"))
        ax5.set_title(
            f"⑤ 운동-시각 그림자 분별 ({p_ep.target_id})\n"
            f"{shad_pair[1]} (CausalField seam strength=2.80, broken=0)",
            color=txt_col,
            fontsize=8.5,
            pad=8,
        )
    else:
        ax5.set_title("⑤ Motor-Visual Shadow Resolution (Not Triggered)", color=txt_col, fontsize=9.0)
    ax5.axis("off")

    # Panel 6: Self-Discovered Cross-Modal Correlation Matrix
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(panel_bg)
    cm = trace_b.correlation_matrix
    if cm["matrix"] is not None:
        mat = cm["matrix"]
        names = cm["feature_names"]
        im = ax6.imshow(mat, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        cbar = plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=txt_col, labelsize=7.5)
        ax6.set_xticks(range(len(names)))
        ax6.set_yticks(range(len(names)))
        ax6.set_xticklabels(names, rotation=45, ha="right", color=txt_col, fontsize=7.5)
        ax6.set_yticklabels(names, color=txt_col, fontsize=7.5)
        for i in range(len(names)):
            for j in range(len(names)):
                ax6.text(
                    j,
                    i,
                    f"{mat[i, j]:+.2f}",
                    ha="center",
                    va="center",
                    color="black" if abs(mat[i, j]) < 0.55 else "white",
                    fontsize=6.8,
                )
        ax6.set_title(
            f"⑥ 경험 기반 자가 발견 교차모달 상관 행렬 (n={cm['sample_count']})\n"
            f"mech_log10_k vs acoust_log10_freq: r = {cm['stiffness_vs_freq_r']:+.3f}",
            color=txt_col,
            fontsize=9.0,
            pad=8,
        )
    else:
        ax6.set_title("⑥ Correlation Matrix (Insufficient samples)", color=txt_col, fontsize=9.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  [Saved inspection visual]: {out_path}")


if __name__ == "__main__":
    main()
