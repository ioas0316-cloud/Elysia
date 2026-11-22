"""
verify_spacetime_structural_cinema.py

Spacetime Structural Cinema Verifier & Visual Renderer
======================================================
Runs SpacetimeStructuralCinema over DynamicSpacetimeWorld (real rotating 4K
Yosemite + second_real_world_4k disks), prints full ingested generators, codec
compression ratio, unseen scrub accuracy (t=0.75, t=2.0), exact Clifford rewind
errors, slow-motion cleavage tau_crit, counterfactual branches, and renders:
  - c:/Elysia/docs/assets/spacetime_structural_cinema_inspection.png (6 panels)
  - c:/Elysia/docs/assets/spacetime_structural_cinema.gif (Forward + Exact Rewind)
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

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

from core.sensory.grounded_rotor_spectrum_layer import circular_axis_diff_deg
from core.sensory.spacetime_structural_cinema import (
    SpacetimeCinemaSession,
    SpacetimeStructuralCinema,
)
from simulators.dynamic_spacetime_world import DynamicSpacetimeWorld
from simulators.hidden_multiphysics_world import YOSEMITE_4K_PATH

OUT_PNG_PATH = Path(r"c:\Elysia\docs\assets\spacetime_structural_cinema_inspection.png")
OUT_GIF_PATH = Path(r"c:\Elysia\docs\assets\spacetime_structural_cinema.gif")


def main() -> None:
    print("=" * 94)
    print("  ELYSIA · Spacetime Structural Cinema (Clifford RotorCodec, Scrub, Rewind & What-If)")
    print("=" * 94)

    world = DynamicSpacetimeWorld(YOSEMITE_4K_PATH, seed=42)
    cinema = SpacetimeStructuralCinema()
    session = cinema.ingest(world, sample_times=(0.0, 0.5, 1.0, 1.5))

    print(
        f"\n[1] Playable Structural Codec Compression: "
        f"{session.raw_video_bytes:,} bytes raw sensory arrays -> "
        f"{session.structural_tracks_bytes} bytes Clifford generators "
        f"({session.compression_ratio:.1f}x reduction)"
    )

    print("\n[2] Extracted Clifford Spacetime Tracks (t0 = 0.0s .. 1.5s)")
    for tr in session.tracks:
        gt0 = world.evaluate_ground_truth_at(tr.t0, *tr.pos0_yx_norm)
        gt_name = gt0["body_id"] if gt0 else "UNKNOWN"
        om_deg = float(np.rad2deg(tr.omega_rad_per_s))
        gt_om = gt0["omega_deg_s"] if gt0 else 0.0
        print(
            f"  - {tr.track_id} [{gt_name:18s}] label={tr.prototype_label:30s} | "
            f"omega={om_deg:+6.2f} deg/s (gt={gt_om:+6.1f}) | "
            f"vel=({tr.vel_yx_norm_per_s[0]:+.3f},{tr.vel_yx_norm_per_s[1]:+.3f})/s | "
            f"rho={tr.optical_mech_phase_lock:+.3f} | fy_w={tr.fy_weak_n:.2f} N | f_tap={tr.acoustic_peak_hz:.0f} Hz"
        )

    print("\n[3] Blind Continuous Scrub Validation at UNSEEN Timestamps (t=0.75 interpolated, t=2.0 future)")
    track_map = {t.track_id: t for t in session.tracks}
    for v in session.scrub_validations:
        tr = track_map[v.track_id]
        gt = world.evaluate_ground_truth_at(v.t_test, *v.observed_pos_yx_norm)
        mech_err = (
            f"{circular_axis_diff_deg(v.predicted_phi_mech_deg, gt['phi_mech_deg']):.2f} deg"
            if (gt and tr.is_anisotropic)
            else "N/A(iso)"
        )
        opt_err = f"{v.optical_angle_error_deg:.2f} deg" if v.optical_angle_error_deg is not None else "N/A(iso)"
        print(
            f"  - {v.track_id} [{v.mode:19s} t={v.t_test:.2f}s] cent_err={v.centroid_error_norm:.4f} | "
            f"opt_err={opt_err:9s} | mech_err={mech_err:9s} | "
            f"cleave={str(v.cleavage_confirmed):5s} (flow={v.cleavage_flow_mag:.2f}px)"
        )

    print("\n[4] Exact Clifford Inverse Rewind & Slow-Motion Cleavage Zoom")
    for rw in session.rewind_traces:
        print(
            f"  - Rewind {rw.track_id}: t={rw.t_from:.1f}->{rw.t_to:.1f}s | err={rw.rewind_vector_error:.3e} | "
            f"class={rw.classification} (stab={rw.phase_lock_stability:.3f})"
        )
    for zm in session.slow_motion_zooms:
        print(
            f"  - SlowMo Zoom {zm.track_id} at t={zm.t_eval:.1f}s probe={zm.probe_angle_deg:.1f} deg: "
            f"k_eff={zm.effective_k_n_m:.0f} N/m, fy_eff={zm.effective_fy_n:.2f} N, "
            f"tau_crit={zm.tau_crit:.3f}" if zm.tau_crit else f"tau_crit=None"
        )

    print("\n[5] Counterfactual Reality Forks (Predicted vs Empirical World Verification)")
    for cf in session.counterfactual_branches:
        print(
            f"  - {cf.branch_name:28s} ({cf.track_id}) ang={cf.probe_angle_deg:5.1f} deg: "
            f"pred(k={cf.predicted_k_n_m:6.0f}, fy={cf.predicted_fy_n:4.1f}N, clv={str(cf.predicted_cleavage):5s}) vs "
            f"actual(k={cf.actual_k_n_m:6.0f}, yld={str(cf.actual_yield):5s}, flow={cf.actual_cleavage_flow_mag:.2f}px) | "
            f"MATCH={cf.prediction_matched}"
        )

    field = session.final_causal_field
    print(f"\n[6] Spacetime CausalField: {len(field.voxels)} voxels, {len(field.beams)} beams")
    for k, v in session.temporal_reflection_summary.items():
        print(f"  * {k}: {v}")

    render_inspection_figure(cinema, session, world, OUT_PNG_PATH)
    render_cinema_gif(cinema, session, world, OUT_GIF_PATH)
    print(f"\nSaved inspection figure: {OUT_PNG_PATH}")
    print(f"Saved cinema playback GIF: {OUT_GIF_PATH}")


def render_inspection_figure(
    cinema: SpacetimeStructuralCinema,
    session: SpacetimeCinemaSession,
    world: DynamicSpacetimeWorld,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(22, 13.5), facecolor="#0d1117")
    fig.suptitle(
        "ELYSIA · Spacetime Structural Cinema (Clifford Rotor Codec, Continuous Scrub, Exact Rewind & What-If)",
        color="#f0f6fc",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(
        2, 3, figure=fig, hspace=0.36, wspace=0.23, left=0.05, right=0.97, top=0.91, bottom=0.06
    )
    panel_bg = "#161b22"
    txt_col = "#e6edf3"

    # Panel 1: Spacetime Trajectory & Live Predicted Clifford Rotors on Future t=2.0s Frame
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(panel_bg)
    frame_20 = world.observe_frame_at(2.0)
    ax1.imshow(frame_20)
    h, w = frame_20.shape[:2]
    ts_trail = np.linspace(0.0, 2.0, 17)
    tr_cols = ["#38bdf8", "#f43f5e", "#3fb950"]
    for idx, tr in enumerate(session.tracks):
        col = tr_cols[idx % len(tr_cols)]
        ys = [cinema.evaluate_track_at(tr, t).pos_yx_norm[0] * h for t in ts_trail]
        xs = [cinema.evaluate_track_at(tr, t).pos_yx_norm[1] * w for t in ts_trail]
        ax1.plot(xs, ys, color=col, linestyle="--", linewidth=1.8, alpha=0.9, zorder=3)
        ax1.scatter([xs[0]], [ys[0]], s=70, facecolors="none", edgecolors=col, linewidths=1.8, zorder=4)
        st_2 = cinema.evaluate_track_at(tr, 2.0)
        y2, x2 = st_2.pos_yx_norm[0] * h, st_2.pos_yx_norm[1] * w
        if tr.optical_coherence_mean > 0.25:
            po = np.deg2rad(st_2.phi_opt_deg)
            ax1.plot(
                [x2 - 14 * np.cos(po), x2 + 14 * np.cos(po)],
                [y2 - 14 * np.sin(po), y2 + 14 * np.sin(po)],
                color="#facc15",
                linewidth=2.6,
                zorder=5,
                label="Opt Rotor (t=2s)" if idx == 0 else None,
            )
        if tr.is_anisotropic:
            pm = np.deg2rad(st_2.phi_mech_deg)
            ax1.plot(
                [x2 - 16 * np.cos(pm), x2 + 16 * np.cos(pm)],
                [y2 - 16 * np.sin(pm), y2 + 16 * np.sin(pm)],
                color="#00f5d4",
                linestyle=":",
                linewidth=2.2,
                zorder=6,
                label="Mech Rotor (t=2s)" if idx == 0 else None,
            )
        ax1.text(
            x2,
            y2 + 23.0,
            f"{tr.track_id} ({float(np.rad2deg(tr.omega_rad_per_s)):+.1f}°/s)",
            color="white",
            fontsize=7.2,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="#0d1117", alpha=0.80, edgecolor=col, lw=0.8),
        )
    ax1.legend(facecolor=panel_bg, labelcolor=txt_col, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    ax1.set_title(
        "① Unseen Future t=2.0s Frame + Clifford Trajectory & Rotors\n"
        "Hollow Circle=t=0s | Dashed Trail=t0..2s | Gold=Opt | Cyan=Mech",
        color=txt_col,
        fontsize=9.1,
        pad=8,
    )
    ax1.axis("off")

    # Panel 2: Continuous Scrub Prediction vs Unseen Interpolated (0.75s) & Future (2.0s) Truth
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(panel_bg)
    ax2.axvspan(0.0, 1.5, color="#1f6feb", alpha=0.10, label="Ingested Span (0..1.5s)")
    ax2.axvspan(1.5, 2.05, color="#f43f5e", alpha=0.10, label="Unseen Future Horizon")
    t_dense = np.linspace(0.0, 2.0, 81)
    aniso_tracks = [tr for tr in session.tracks if tr.is_anisotropic]
    cols_2 = ["#38bdf8", "#f43f5e"]
    for idx, tr in enumerate(aniso_tracks):
        col = cols_2[idx % len(cols_2)]
        # Unwrap continuous curve around 0..180 nicely for display via mod 180 NaN breaks or linear angle
        phi_pred_raw = [
            float(np.rad2deg(tr.phi_mech0_rad + tr.omega_rad_per_s * (t - tr.t0)) % 180.0)
            for t in t_dense
        ]
        # Insert NaN breaks where modular wrap jumps > 90 deg for clean visual plotting
        t_plot, y_plot = [], []
        for k, val in enumerate(phi_pred_raw):
            if k > 0 and abs(val - phi_pred_raw[k - 1]) > 90.0:
                t_plot.append(t_dense[k])
                y_plot.append(np.nan)
            t_plot.append(t_dense[k])
            y_plot.append(val)
        gt0 = world.evaluate_ground_truth_at(tr.t0, *tr.pos0_yx_norm)
        lbl = gt0["body_id"] if gt0 else tr.track_id
        ax2.plot(t_plot, y_plot, color=col, linewidth=2.2, label=f"{tr.track_id} ({lbl[:14]})")

        for t_unseen in (0.75, 2.0):
            st = cinema.evaluate_track_at(tr, t_unseen)
            gt_u = world.evaluate_ground_truth_at(t_unseen, *st.pos_yx_norm)
            if gt_u:
                ax2.scatter([t_unseen], [gt_u["phi_mech_deg"]], marker="*", s=150, color="#facc15", edgecolors="white", zorder=6)
                ax2.scatter([t_unseen], [st.phi_mech_deg], marker="o", s=55, color=col, edgecolors="white", zorder=7)

    ax2.scatter([], [], marker="*", s=120, color="#facc15", edgecolors="white", label="Unseen GT Bond Angle")
    ax2.set_xlabel("Continuous Time t [s]", color=txt_col, fontsize=8.5)
    ax2.set_ylabel("Internal Bond Axis phi_mech(t) [deg]", color=txt_col, fontsize=8.5)
    ax2.set_ylim(-5, 185)
    ax2.tick_params(colors=txt_col, labelsize=8)
    ax2.grid(color="#30363d", linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.legend(facecolor=panel_bg, labelcolor=txt_col, fontsize=7.0, loc="lower left", ncol=2)
    ax2.set_title(
        "② Continuous Scrub Trajectory vs Unseen Ground Truth\n"
        "Evaluated Without World Peeking at t=0.75s (Interp) & t=2.0s (Future)",
        color=txt_col,
        fontsize=9.1,
        pad=8,
    )

    # Panel 3: Slow-Motion Cleavage Zoom around tau_crit
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(panel_bg)
    if session.slow_motion_zooms:
        zm = session.slow_motion_zooms[0]
        taus = [f.tau for f in zm.frames]
        disps = [f.disp_mm for f in zm.frames]
        gaps = [f.plastic_gap_mm for f in zm.frames]
        ax3.plot(taus, disps, color="#38bdf8", linewidth=2.2, label="Total Indentation [mm]")
        ax3.plot(taus, gaps, color="#f43f5e", linewidth=2.2, label="Irreversible Plastic Cleavage Gap [mm]")
        if zm.tau_crit is not None:
            ax3.axvline(zm.tau_crit, color="#facc15", linestyle="--", linewidth=1.6, label=f"Critical Yield tau_crit={zm.tau_crit:.3f}")
            ax3.axvspan(zm.tau_crit, 1.0, color="#f43f5e", alpha=0.10)
    ax3.set_xlabel("Normalized Slow-Motion Load Time tau in [0, 1]", color=txt_col, fontsize=8.5)
    ax3.set_ylabel("Opening / Displacement [mm]", color=txt_col, fontsize=8.5)
    ax3.tick_params(colors=txt_col, labelsize=8)
    ax3.grid(color="#30363d", linestyle="--", linewidth=0.5, alpha=0.6)
    ax3.legend(facecolor=panel_bg, labelcolor=txt_col, fontsize=7.3, loc="upper left")
    ax3.set_title(
        "③ Slow-Motion Cleavage Zoom (24 Continuous Subframes)\n"
        "Pinpointing Critical Fracture Onset tau_crit = Fy / F_applied",
        color=txt_col,
        fontsize=9.1,
        pad=8,
    )

    # Panel 4: Counterfactual Reality Forks (Predicted vs Actual k & Cleavage at t=1.0 & t=2.0)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(panel_bg)
    first_tr_cfs = session.counterfactual_branches[:4]
    short_names = [
        "BrA:Weak\nt=1.0s",
        "BrB:Strong\nt=1.0s",
        "BrC:Stale t0\nt=2.0s",
        "BrC:Rotor\nt=2.0s",
    ]
    x_idx = np.arange(len(first_tr_cfs))
    w_bar = 0.34
    pred_ks = [c.predicted_k_n_m / 1000.0 for c in first_tr_cfs]
    act_ks = [(c.actual_k_n_m or 0.0) / 1000.0 for c in first_tr_cfs]
    ax4.bar(x_idx - w_bar / 2, pred_ks, width=w_bar, color="#38bdf8", label="Predicted Stiffness [kN/m]")
    act_cols = ["#f43f5e" if c.actual_yield else "#3fb950" for c in first_tr_cfs]
    ax4.bar(x_idx + w_bar / 2, act_ks, width=w_bar, color=act_cols, label="Actual (Red=Cleaved, Green=Held)")
    for xi, c, ak in zip(x_idx, first_tr_cfs, act_ks):
        badge = "CLEAVE" if c.actual_yield else "HOLD"
        col_b = "#f43f5e" if c.actual_yield else "#3fb950"
        ax4.text(xi, max(ak, pred_ks[xi]) + 0.5, badge, color=col_b, fontsize=7.6, fontweight="bold", ha="center")
    ax4.set_xticks(x_idx)
    ax4.set_xticklabels(short_names, color=txt_col, fontsize=8.0)
    ax4.set_ylabel("Directional Stiffness [kN/m]", color=txt_col, fontsize=8.5)
    ax4.set_ylim(0, max(pred_ks + act_ks + [12.0]) * 1.25)
    ax4.tick_params(colors=txt_col, labelsize=8)
    ax4.grid(axis="y", color="#30363d", linestyle="--", linewidth=0.5, alpha=0.6)
    ax4.legend(facecolor=panel_bg, labelcolor=txt_col, fontsize=7.3, loc="upper left")
    ax4.set_title(
        "④ Counterfactual Video Forking vs Actual Physical World\n"
        "Stale t=0 Angle Fails at t=2.0s While Clifford Rotor Cleaves Cleanly",
        color=txt_col,
        fontsize=9.1,
        pad=8,
    )

    # Panel 5: Exact Clifford Rewind Invertibility & Codec Compression Metrics
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(panel_bg)
    rw_names = [f"{rw.track_id}\n({rw.classification[:15]})" for rw in session.rewind_traces]
    # Plot -log10(max(err, 1e-16)) showing > 12 orders of magnitude exact preservation
    neg_log_errs = [-np.log10(max(rw.rewind_vector_error, 1e-16)) for rw in session.rewind_traces]
    y_rw = np.arange(len(rw_names))
    ax5.barh(y_rw, neg_log_errs, height=0.45, color="#a855f7", edgecolor="white", linewidth=0.9)
    ax5.axvline(12.0, color="#facc15", linestyle="--", linewidth=1.4, label="Machine Precision Floor (<1e-12)")
    for yi, nle, rw in zip(y_rw, neg_log_errs, session.rewind_traces):
        ax5.text(min(nle - 0.6, 15.2), yi, f"err={rw.rewind_vector_error:.1e}", color="white", va="center", ha="right", fontsize=7.8, fontweight="bold")
    ax5.set_yticks(y_rw)
    ax5.set_yticklabels(rw_names, color=txt_col, fontsize=7.8)
    ax5.set_xlim(0.0, 17.5)
    ax5.set_ylim(-1.1, len(rw_names) + 0.35)
    ax5.set_xlabel("Algebraic Rewind Fidelity (-log10 Vector Error)", color=txt_col, fontsize=8.5)
    ax5.tick_params(colors=txt_col, labelsize=8)
    ax5.grid(axis="x", color="#30363d", linestyle="--", linewidth=0.5, alpha=0.6)
    ax5.legend(facecolor=panel_bg, labelcolor=txt_col, fontsize=7.3, loc="upper right")
    ax5.text(
        0.5,
        -0.95,
        f"Structural Codec Ratio: {session.raw_video_bytes:,} B raw -> {session.structural_tracks_bytes} B generators ({session.compression_ratio:.1f}x)",
        color="#facc15",
        fontsize=8.2,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117", edgecolor="#30363d"),
    )
    ax5.set_title(
        "⑤ Bit-Exact Clifford Rewind (R~ v' R) & Principle Compression\n"
        "Separates Intrinsic Cross-Ply Veneer vs Phase-Locked Cohesive Timber",
        color=txt_col,
        fontsize=9.1,
        pad=8,
    )

    # Panel 6: Multi-Scale Spacetime CausalField (L2 -> L8 -> L10 -> L12 -> L14)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(panel_bg)
    field = session.final_causal_field
    tiers = [
        ("L14 Rewind & Counterfactual Forks", [k for k in field.voxels if k.startswith(("rewind_", "counterfactual_"))], "#a855f7", 0.88),
        ("L12 Temporal 4-Axis Reflection", [k for k in field.voxels if k.startswith("reflection_")], "#f43f5e", 0.70),
        ("L10 Clifford Generators", [k for k in field.voxels if k.startswith("generator_")], "#facc15", 0.52),
        ("L8 Spacetime Tracks", [k for k in field.voxels if k.startswith("track_")], "#38bdf8", 0.34),
        ("L2 Optical Regions", [k for k in field.voxels if k.startswith("region_")], "#3fb950", 0.16),
    ]
    pos_map = {}
    for t_lbl, keys, col, yv in tiers:
        ax6.axhline(yv, color="#21262d", linewidth=0.9, zorder=1)
        ax6.text(0.01, yv + 0.04, f"{t_lbl} ({len(keys)})", color=col, fontsize=7.6, fontweight="bold")
        xs_t = np.linspace(0.22, 0.92, len(keys)) if len(keys) > 1 else [0.57]
        for xv, k in zip(xs_t, keys):
            pos_map[k] = (float(xv), float(yv))
            ax6.scatter([xv], [yv], s=105, color=col, edgecolors="white", linewidths=1.0, zorder=4)
            short = (
                k.replace("counterfactual_", "cf:")
                .replace("rewind_", "rw:")
                .replace("generator_", "gen:")
                .replace("track_", "trk:")
                .replace("reflection_temporal_4axis", "refl_4axis")
            )
            ax6.text(xv, yv - 0.045, short, color=txt_col, fontsize=6.4, ha="center", va="top")

    for beam in field.beams:
        if beam.source_id in pos_map and beam.target_id in pos_map:
            x0, y0 = pos_map[beam.source_id]
            x1, y1 = pos_map[beam.target_id]
            ax6.plot([x0, x1], [y0, y1], color="#64748b", alpha=0.42, linewidth=0.85, zorder=2)

    ax6.set_xlim(0.0, 1.0)
    ax6.set_ylim(0.04, 1.02)
    ax6.set_title(
        f"⑥ Multi-Scale Spacetime CausalField ({len(field.voxels)} Voxels, {len(field.beams)} Beams)\n"
        "Regions (L2) -> Tracks (L8) -> Clifford Generators (L10) -> Temporal Forks (L14)",
        color=txt_col,
        fontsize=9.1,
        pad=8,
    )
    ax6.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_cinema_gif(
    cinema: SpacetimeStructuralCinema,
    session: SpacetimeCinemaSession,
    world: DynamicSpacetimeWorld,
    out_path: Path,
) -> None:
    """Render Forward Play (0..2s) + Cleavage Flash + Exact Clifford Rewind (2..0s)."""
    t_fwd = list(np.linspace(0.0, 2.0, 13))
    t_bwd = list(np.linspace(2.0, 0.0, 13))

    schedule = [(float(t), "FORWARD PLAY  |  R(w*dt) v R~", False) for t in t_fwd]
    schedule.append((2.0, "WEAK-AXIS CAUSAL CLEAVAGE  @  t=2.00s", True))
    schedule.append((2.0, "WEAK-AXIS CAUSAL CLEAVAGE  @  t=2.00s", True))
    schedule.extend([(float(t), "EXACT CLIFFORD REWIND  |  R~ v' R", False) for t in t_bwd])

    frames_rgb = []
    out_w, out_h = 640, 360
    sx = out_w / float(world.CANVAS_W)
    sy = out_h / float(world.CANVAS_H)

    for t_val, banner, show_cleave in schedule:
        cleave_id = "rotating_timber_A" if show_cleave else None
        full = world.render_full_canvas_at(t_val, cleave_body_id=cleave_id)
        small = cv2.resize(full, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # Draw live tracked Clifford rotor vectors on top
        snap = cinema.scrub_to(session.tracks, t_val)
        for tr, st in zip(session.tracks, snap.states):
            cy = int(round(st.pos_yx_norm[0] * out_h))
            cx = int(round(st.pos_yx_norm[1] * out_w))
            if tr.optical_coherence_mean > 0.25:
                po = np.deg2rad(st.phi_opt_deg)
                dx_o, dy_o = int(round(36 * np.cos(po))), int(round(36 * np.sin(po)))
                cv2.line(small, (cx - dx_o, cy - dy_o), (cx + dx_o, cy + dy_o), (250, 204, 21), 3, cv2.LINE_AA)
            if tr.is_anisotropic:
                pm = np.deg2rad(st.phi_mech_deg)
                dx_m, dy_m = int(round(42 * np.cos(pm))), int(round(42 * np.sin(pm)))
                cv2.line(small, (cx - dx_m, cy - dy_m), (cx + dx_m, cy + dy_m), (56, 189, 248), 2, cv2.LINE_AA)
            cv2.circle(small, (cx, cy), 4, (255, 255, 255), -1)

        # Top telemetry ribbon + timeline scrub bar
        cv2.rectangle(small, (0, 0), (out_w, 34), (13, 17, 23), -1)
        col_txt = (244, 63, 94) if show_cleave else ((250, 204, 21) if "REWIND" in banner else (56, 189, 248))
        cv2.putText(
            small,
            f"{banner}   t={t_val:.2f}s",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            col_txt,
            1,
            cv2.LINE_AA,
        )
        bar_x = int(round((t_val / 2.0) * (out_w - 24))) + 12
        cv2.line(small, (12, out_h - 12), (out_w - 12, out_h - 12), (60, 68, 78), 3)
        cv2.circle(small, (bar_x, out_h - 12), 6, col_txt, -1)
        frames_rgb.append(small)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames_rgb, duration=0.125, loop=0)


if __name__ == "__main__":
    main()
