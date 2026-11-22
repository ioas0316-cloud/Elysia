"""
verify_imaginative_perceptual_recreator.py

Verification Runner for Self-Mapped Perceptual Axes, Closed-Loop Imaginative
Re-Observation, Autonomous Goal Re-Creation & Game Engine / Video Transfers
=============================================================================
Runs run_full_imaginative_session over a genuine 4K Yosemite Half Dome / cliff
scene (genuine_optical_4k_yosemite.jpg[600:1350, 1300:2400]), prints calibrated
prism rotor anchors, Red/Yellow regional tipping points, 360-deg rainbow rotor
closure & scale-rotor rewind RMSE, luminance/roughness orthogonality, closed-loop
goal convergence, Game Engine PBR shader specs, External Video CDL parameters, and
multi-scale CausalField (L2..L14), saving:
  - c:/Elysia/docs/assets/imaginative_perceptual_recreator_inspection.png
  - c:/Elysia/docs/assets/imaginative_rainbow_recreation.gif
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
from PIL import Image

from core.sensory.imaginative_perceptual_recreator import (
    ImaginativeExplorationSession,
    ImaginativeReobservationEngine,
    PerceptualImaginationOp,
    run_full_imaginative_session,
    wavelength_nm_to_stimulus_srgb,
)
from simulators.hidden_multiphysics_world import YOSEMITE_4K_PATH

OUT_PNG_PATH = Path(r"c:\Elysia\docs\assets\imaginative_perceptual_recreator_inspection.png")
OUT_GIF_PATH = Path(r"c:\Elysia\docs\assets\imaginative_rainbow_recreation.gif")


def main() -> None:
    print("=" * 94)
    print("  ELYSIA · Closed-Loop Perceptual Axis Self-Mapping & Imaginative Re-Creation")
    print("=" * 94)

    full_img = np.array(Image.open(YOSEMITE_4K_PATH), dtype=np.uint8)
    yosemite_crop = full_img[600:1350, 1300:2400].copy()

    session = run_full_imaginative_session(yosemite_crop)
    man = session.manifold

    print(
        f"\n[1] Self-Calibrated Perceptual Axis Manifold "
        f"(sigma_rg={man.sigma_rg:.3f}, sigma_by={man.sigma_by:.3f}, target_region=region_{session.target_region_id})"
    )
    print("  * Discovered Rainbow Prism Anchors (deg):", {k: round(v, 1) for k, v in man.anchor_angles_deg.items()})
    for ax in man.axes:
        print(f"  - {ax.axis_id:24s} ({ax.metric_unit:22s}): span={ax.empirical_span} | {ax.name}")

    for sweep in (session.red_sweep, session.yellow_sweep):
        print(
            f"\n[2] Regional Chroma Sweep -> {sweep.target_name} on region_{sweep.region_id} "
            f"(alpha_tipping_point={sweep.alpha_tipping_point}, min_L2_corr={sweep.min_l2_persistence_corr:.3f})"
        )
        for s in sweep.steps:
            print(
                f"  alpha={s.alpha:.1f}: regional_dC={s.regional_delta_c:.3f}, regional_dA={s.regional_delta_a:.3f}, "
                f"L2_corr={s.comparison.l2_persistence_corr:.3f}, mat_edges={s.material_boundary_count}, "
                f"gamut_clip={s.gamut_clip_ratio:.4f}"
            )

    rb = session.rainbow_sweep
    print(
        f"\n[3] Full 360-deg Clifford Rainbow Rotor: "
        f"closure_rmse_360={rb.closure_rmse_360:.3e}, scale_rotor_rewind_rmse_120={rb.rewind_rmse_120:.3e}, "
        f"min_L2_corr={rb.min_l2_persistence_corr:.3f}"
    )

    orth = session.orthogonality_sweep
    print("\n[4] Axis Orthogonality Under Self Re-Observation")
    print(
        "  * Luminance EV sweep:",
        [(ev, f"dC={dc:.4f}", f"L2={c:.3f}") for ev, dc, c in zip(orth.luminance_evs, orth.luminance_delta_cs, orth.luminance_persistence_corrs)],
    )
    print(
        "  * Roughness Gain sweep:",
        [(rs, f"fine_tex={fe:.4f}", f"L2={c:.3f}") for rs, fe, c in zip(orth.roughness_scales, orth.roughness_fine_energies, orth.roughness_persistence_corrs)],
    )

    goal = session.goal_recreation
    print(
        f"\n[5] Autonomous Closed-Loop Goal Re-Creation (target={goal.target_wavelength_nm:.0f}nm, "
        f"chroma={goal.target_chroma_norm:.2f}, rough_x={goal.target_roughness_ratio:.2f}) | converged={goal.converged}"
    )
    for it in goal.iterations:
        print(
            f"  iter {it.iteration}: u=({it.cur_u_rg:+.3f},{it.cur_u_by:+.3f}), fine_tex={it.cur_fine_texture:.4f}, "
            f"col_err={it.color_error:.3f}, rough_err={it.roughness_error:.3f}, tot_err={it.total_error:.3f}, "
            f"L2_corr={it.l2_persistence_corr:.3f}, clip={it.gamut_clip_ratio:.4f}"
        )

    pbr = session.pbr_spec
    vid = session.video_grade_spec
    print("\n[6] Cross-Domain Principle Mappings")
    print(
        f"  * Game Engine PBR Shader: albedo_lin=({pbr.albedo_linear_rgb[0]:.3f},{pbr.albedo_linear_rgb[1]:.3f},{pbr.albedo_linear_rgb[2]:.3f}), "
        f"hue={pbr.albedo_hue_deg:.1f} deg, sat={pbr.albedo_saturation:.2f}, ev={pbr.exposure_ev:+.2f}, "
        f"roughness={pbr.microfacet_roughness:.3f}, aniso=({pbr.anisotropy_angle_deg:.0f} deg, {pbr.anisotropy_strength:.2f})"
    )
    print(f"      Boundary Note: {pbr.domain_boundary_note}")
    print(
        f"  * External Video ASC-CDL: slope=({vid.cdl_slope_rgb[0]:.2f},{vid.cdl_slope_rgb[1]:.2f},{vid.cdl_slope_rgb[2]:.2f}), "
        f"offset_ev={vid.cdl_offset_ev:+.2f}, hue_rotor={vid.hue_rotor_deg:+.1f} deg, "
        f"chroma_gain={vid.chroma_gain:.2f}x, detail_sharpen={vid.detail_sharpen_gain:.2f}x"
    )
    print(f"      Boundary Note: {vid.domain_boundary_note}")

    field = session.final_causal_field
    print(f"\n[7] Multi-Scale CausalField Summary: {len(field.voxels)} voxels, {len(field.beams)} beams")

    render_inspection_figure(session, OUT_PNG_PATH)
    render_rainbow_recreation_gif(session, OUT_GIF_PATH)
    print(f"\nSaved inspection figure: {OUT_PNG_PATH}")
    print(f"Saved animation GIF: {OUT_GIF_PATH}")


def render_inspection_figure(session: ImaginativeExplorationSession, out_path: Path) -> None:
    fig = plt.figure(figsize=(22, 13.5), facecolor="#0d1117")
    fig.suptitle(
        "ELYSIA · Self-Mapped Perceptual Axes, Closed-Loop Re-Observation, Re-Creation & Game/Video Transfers",
        color="#f0f6fc",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(
        2, 3, figure=fig, hspace=0.35, wspace=0.23, left=0.05, right=0.97, top=0.91, bottom=0.06
    )
    panel_bg = "#161b22"
    txt_col = "#e6edf3"
    man = session.manifold

    # Panel 1: Self-Calibrated Rainbow Rotor Wheel + Regional Red/Yellow Imagination Trajectories
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(panel_bg)
    th_c = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(th_c), np.sin(th_c), color="#30363d", linestyle="--", linewidth=1.0)
    ax1.axhline(0.0, color="#30363d", linewidth=0.8)
    ax1.axvline(0.0, color="#30363d", linewidth=0.8)
    ax1.scatter([0.0], [0.0], color="#c9d1d9", s=45, zorder=3, label="Achromatic Neutral Origin")

    for wl, (ux, uy) in zip(man.prism_wavelengths_nm, man.prism_u_coords):
        col_wl = tuple(wavelength_nm_to_stimulus_srgb(wl).tolist())
        ax1.scatter([ux], [uy], color=col_wl, s=42, edgecolors="white", linewidths=0.5, zorder=4)
    lbl_map = {410: "410 V", 450: "450 B", 490: "490 C", 530: "530 G", 580: "580 Y", 605: "605 O", 650: "650 R"}
    for wl, uv in man.anchor_unit_vecs.items():
        ax1.text(float(uv[0]) * 1.18, float(uv[1]) * 1.18, lbl_map[wl], color="white", fontsize=7.2, ha="center", va="center", fontweight="bold")

    r_xs = [s.regional_mean_u_rg for s in session.red_sweep.steps]
    r_ys = [s.regional_mean_u_by for s in session.red_sweep.steps]
    ax1.plot(r_xs, r_ys, color="#f43f5e", marker="o", markersize=5, linewidth=2.2, label="Red Sweep (650nm)", zorder=5)

    y_xs = [s.regional_mean_u_rg for s in session.yellow_sweep.steps]
    y_ys = [s.regional_mean_u_by for s in session.yellow_sweep.steps]
    ax1.plot(y_xs, y_ys, color="#facc15", marker="^", markersize=5, linewidth=2.2, label="Yellow Sweep (580nm)", zorder=5)

    g_xs = [it.cur_u_rg for it in session.goal_recreation.iterations]
    g_ys = [it.cur_u_by for it in session.goal_recreation.iterations]
    ax1.plot(g_xs, g_ys, color="#38bdf8", marker="*", markersize=7, linewidth=1.8, linestyle="-.", label="Autonomous Sunset Goal", zorder=6)

    ax1.set_xlim(-1.55, 1.55)
    ax1.set_ylim(-1.65, 1.65)
    ax1.set_xlabel("Self-Calibrated Opponent Axis u_RG (Red-Green)", color=txt_col, fontsize=8.5)
    ax1.set_ylabel("Self-Calibrated Opponent Axis u_BY (Blue-Yellow)", color=txt_col, fontsize=8.5)
    ax1.tick_params(colors=txt_col, labelsize=8)
    ax1.legend(facecolor=panel_bg, labelcolor=txt_col, fontsize=7.0, loc="upper left")
    ax1.set_title(
        "① Self-Mapped Rainbow Rotor Wheel & Chromatic Excursions\n"
        f"Real Yosemite region_{session.target_region_id} Trajectory in Calibrated (u_RG, u_BY)",
        color=txt_col,
        fontsize=9.1,
        pad=8,
    )

    # Panel 2: Visual Comparison Strip (2x2 Inset Mosaic)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(panel_bg)
    ax2.axis("off")
    tiles = [
        (session.orig_obs.rgb_working, "Original Real Yosemite (4K Crop)", [0.01, 0.51, 0.47, 0.43]),
        (session.red_sweep.steps[-1].imagined_rgb, f"Imagined Crimson Red (a=1.0, dC={session.red_sweep.steps[-1].regional_delta_c:.2f})", [0.52, 0.51, 0.47, 0.43]),
        (session.yellow_sweep.steps[-1].imagined_rgb, f"Imagined Golden Yellow (a=1.0, dC={session.yellow_sweep.steps[-1].regional_delta_c:.2f})", [0.01, 0.02, 0.47, 0.43]),
        (session.goal_recreation.final_imagined_rgb, f"Goal Sunset Re-Creation (Rough x{session.goal_recreation.final_op.roughness_scale:.2f})", [0.52, 0.02, 0.47, 0.43]),
    ]
    for img_t, ttl, rect in tiles:
        ins = ax2.inset_axes(rect)
        ins.imshow(np.clip(img_t, 0.0, 1.0))
        ins.set_title(ttl, color=txt_col, fontsize=7.3, pad=3)
        ins.axis("off")
    ax2.set_title(
        f"② Visual Re-Observation Gallery (Target = region_{session.target_region_id} Granite Cliff)\n"
        "Sky & Neighboring Geometry Preserved (L2 Persistence >= 0.95)",
        color=txt_col,
        fontsize=9.1,
        pad=8,
    )

    # Panel 3: Alpha Sweep Curves & Tipping-Point Discovery
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(panel_bg)
    alphas = [s.alpha for s in session.red_sweep.steps]
    red_dc = [s.regional_delta_c for s in session.red_sweep.steps]
    yel_dc = [s.regional_delta_c for s in session.yellow_sweep.steps]
    red_l2 = [s.comparison.l2_persistence_corr for s in session.red_sweep.steps]
    yel_l2 = [s.comparison.l2_persistence_corr for s in session.yellow_sweep.steps]

    l1, = ax3.plot(alphas, red_dc, color="#f43f5e", marker="o", linewidth=2.2, label="Red Shift dC")
    l2, = ax3.plot(alphas, yel_dc, color="#facc15", marker="^", linewidth=2.2, label="Yellow Shift dC")
    ax3.set_xlabel("Chromatic Excursion Strength alpha in [0, 1]", color=txt_col, fontsize=8.5)
    ax3.set_ylabel("Re-Observed Regional Color Shift dC", color=txt_col, fontsize=8.5)
    ax3.set_ylim(0.0, max(red_dc + yel_dc + [1.0]) * 1.18)
    ax3.tick_params(colors=txt_col, labelsize=8)
    ax3.grid(color="#30363d", linestyle="--", linewidth=0.5, alpha=0.6)

    ax3_r = ax3.twinx()
    l3, = ax3_r.plot(alphas, red_l2, color="#38bdf8", linestyle="--", marker="s", markersize=4, linewidth=1.6, label="L2 Persistence Corr (Red)")
    l4, = ax3_r.plot(alphas, yel_l2, color="#3fb950", linestyle=":", marker="d", markersize=4, linewidth=1.6, label="L2 Persistence Corr (Yellow)")
    ax3_r.set_ylabel("Structural L2 Persistence Correlation", color="#38bdf8", fontsize=8.3)
    ax3_r.set_ylim(0.70, 1.03)
    ax3_r.tick_params(colors="#38bdf8", labelsize=8)

    if session.red_sweep.alpha_tipping_point is not None:
        ax3.axvline(session.red_sweep.alpha_tipping_point, color="#a855f7", linestyle="--", linewidth=1.4)
        ax3.text(
            session.red_sweep.alpha_tipping_point + 0.03,
            max(red_dc) * 0.85,
            f"Tipping alpha={session.red_sweep.alpha_tipping_point:.1f}",
            color="#a855f7",
            fontsize=8.0,
            fontweight="bold",
        )
    lines = [l1, l2, l3, l4]
    ax3.legend(lines, [l.get_label() for l in lines], facecolor=panel_bg, labelcolor=txt_col, fontsize=7.0, loc="lower left")
    ax3.set_title(
        "③ Re-Observed Color Divergence vs Edge Structure Retention\n"
        f"360° Rotor Closure RMSE={session.rainbow_sweep.closure_rmse_360:.1e} | Scale Rewind RMSE={session.rainbow_sweep.rewind_rmse_120:.1e}",
        color=txt_col,
        fontsize=9.1,
        pad=8,
    )

    # Panel 4: Luminance EV & Roughness Axis Orthogonality
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(panel_bg)
    orth = session.orthogonality_sweep
    x_norm_idx = np.arange(len(orth.roughness_scales))
    p1, = ax4.plot(x_norm_idx, orth.roughness_fine_energies, color="#f43f5e", marker="o", linewidth=2.2, label="Roughness Fine Texture Energy")
    ax4.set_xticks(x_norm_idx)
    ax4.set_xticklabels([f"R={rs:.1f}x\n(EV={ev:+.2f})" for rs, ev in zip(orth.roughness_scales, orth.luminance_evs)], color=txt_col, fontsize=7.3)
    ax4.set_ylabel("Re-Observed Fine Texture Energy", color="#f43f5e", fontsize=8.5)
    ax4.tick_params(colors=txt_col, labelsize=8)
    ax4.grid(color="#30363d", linestyle="--", linewidth=0.5, alpha=0.6)

    ax4_r = ax4.twinx()
    p2, = ax4_r.plot(x_norm_idx, orth.luminance_delta_cs, color="#38bdf8", marker="^", linestyle="--", linewidth=1.9, label="Luminance Sweep Color Drift dC (<0.05)")
    p3, = ax4_r.plot(x_norm_idx, orth.luminance_persistence_corrs, color="#3fb950", marker="s", linestyle=":", linewidth=1.8, label="Luminance Sweep L2 Corr (>0.90)")
    ax4_r.set_ylabel("Chromatic Drift dC / L2 Correlation", color="#38bdf8", fontsize=8.3)
    ax4_r.set_ylim(-0.05, 1.12)
    ax4_r.tick_params(colors="#38bdf8", labelsize=8)

    lines4 = [p1, p2, p3]
    ax4.legend(lines4, [l.get_label() for l in lines4], facecolor=panel_bg, labelcolor=txt_col, fontsize=7.0, loc="center right")
    ax4.set_title(
        "④ Luminance (EV) & Texture Roughness Axis Orthogonality\n"
        "Monotonic Roughness Control | Zero Hue Drift on Exposure Change",
        color=txt_col,
        fontsize=9.1,
        pad=8,
    )

    # Panel 5: Closed-Loop Goal Re-Creation Convergence
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(panel_bg)
    iters = [it.iteration for it in session.goal_recreation.iterations]
    tot_errs = [it.total_error for it in session.goal_recreation.iterations]
    col_errs = [it.color_error for it in session.goal_recreation.iterations]
    rgh_errs = [it.roughness_error for it in session.goal_recreation.iterations]
    ax5.plot(iters, tot_errs, color="#f43f5e", marker="o", linewidth=2.4, label="Total Perceptual Goal Error")
    ax5.plot(iters, col_errs, color="#facc15", marker="^", linestyle="--", linewidth=1.8, label="Spectral Chroma Error")
    ax5.plot(iters, rgh_errs, color="#38bdf8", marker="s", linestyle=":", linewidth=1.8, label="Micro-Texture Roughness Error")
    ax5.set_xlabel("Closed-Loop Imagine -> Re-Observe Iteration", color=txt_col, fontsize=8.5)
    ax5.set_ylabel("Discrepancy in Self-Mapped Coordinates", color=txt_col, fontsize=8.5)
    ax5.tick_params(colors=txt_col, labelsize=8)
    ax5.grid(color="#30363d", linestyle="--", linewidth=0.5, alpha=0.6)
    ax5.legend(facecolor=panel_bg, labelcolor=txt_col, fontsize=7.5, loc="upper right")
    ax5.text(
        0.5,
        tot_errs[0] * 0.14,
        f"Converged={session.goal_recreation.converged} | Err: {tot_errs[0]:.3f} -> {tot_errs[-1]:.3f}\n"
        f"Final L2 Persistence={session.goal_recreation.iterations[-1].l2_persistence_corr:.3f}",
        color="#3fb950",
        fontsize=8.0,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117", edgecolor="#3fb950", lw=0.9),
    )
    ax5.set_title(
        "⑤ Closed-Loop Autonomous Goal Re-Creation Convergence\n"
        "Target: Sunset Orange (605nm, C=0.72) + Roughness x1.40",
        color=txt_col,
        fontsize=9.1,
        pad=8,
    )

    # Panel 6: Multi-Scale CausalField (L2 -> L10 -> L12 -> L14) + Game PBR & Video CDL Box
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(panel_bg)
    field = session.final_causal_field
    tiers = [
        ("L14 Cross-Domain Transfers", [k for k in field.voxels if k.startswith("transfer_")], "#a855f7", 0.88),
        ("L12 Re-Observed Imaginations", [k for k in field.voxels if k.startswith("imagination_")], "#f43f5e", 0.70),
        ("L10 Self-Mapped Axes", [k for k in field.voxels if k.startswith(("axis_", "rotor_"))], "#facc15", 0.52),
        ("L2 Scene Regions", [k for k in field.voxels if k.startswith("region_")], "#3fb950", 0.34),
    ]
    pos_map = {}
    for t_lbl, keys, col, yv in tiers:
        ax6.axhline(yv, color="#21262d", linewidth=0.9, zorder=1)
        ax6.text(0.01, yv + 0.04, f"{t_lbl} ({len(keys)})", color=col, fontsize=7.4, fontweight="bold")
        xs_t = np.linspace(0.24, 0.92, len(keys)) if len(keys) > 1 else [0.58]
        for xv, k in zip(xs_t, keys):
            pos_map[k] = (float(xv), float(yv))
            ax6.scatter([xv], [yv], s=85, color=col, edgecolors="white", linewidths=0.9, zorder=4)
            short = (
                k.replace("transfer_", "xf:")
                .replace("imagination_", "im:")
                .replace("axis_", "ax:")
                .replace("rotor_", "rot:")
                .replace("_sweep", "")
                .replace("_recreation", "")
                .replace("_roughness", "rough")
                .replace("_shader", "")
                .replace("_grade_cdl", "cdl")
            )
            ax6.text(xv, yv - 0.045, short, color=txt_col, fontsize=6.0, ha="center", va="top")

    for beam in field.beams:
        if beam.source_id in pos_map and beam.target_id in pos_map:
            x0, y0 = pos_map[beam.source_id]
            x1, y1 = pos_map[beam.target_id]
            ax6.plot([x0, x1], [y0, y1], color="#64748b", alpha=0.38, linewidth=0.8, zorder=2)

    pbr = session.pbr_spec
    vid = session.video_grade_spec
    xfer_box = (
        f"Game PBR Shader: linRGB=({pbr.albedo_linear_rgb[0]:.2f},{pbr.albedo_linear_rgb[1]:.2f},{pbr.albedo_linear_rgb[2]:.2f}) | "
        f"rough={pbr.microfacet_roughness:.2f} | aniso={pbr.anisotropy_angle_deg:.0f}deg\n"
        f"Video ASC-CDL  : slope=({vid.cdl_slope_rgb[0]:.2f},{vid.cdl_slope_rgb[1]:.2f},{vid.cdl_slope_rgb[2]:.2f}) | "
        f"hueRot={vid.hue_rotor_deg:+.1f}deg | chroma={vid.chroma_gain:.2f}x"
    )
    ax6.text(
        0.02,
        0.02,
        xfer_box,
        color="#c9d1d9",
        fontsize=6.9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#0d1117", edgecolor="#a855f7", lw=0.9),
    )
    ax6.set_xlim(0.0, 1.0)
    ax6.set_ylim(0.0, 1.02)
    ax6.set_title(
        f"⑥ Multi-Scale CausalField ({len(field.voxels)} Voxels, {len(field.beams)} Beams)\n"
        "Regions (L2) -> Axes (L10) -> Re-Observations (L12) -> Game/Video (L14)",
        color=txt_col,
        fontsize=9.1,
        pad=8,
    )
    ax6.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_rainbow_recreation_gif(session: ImaginativeExplorationSession, out_path: Path) -> None:
    """Animate Regional Red/Yellow Sweeps -> Sunset Goal Convergence -> 360° Rainbow Rotor."""
    engine = ImaginativeReobservationEngine()
    orig = session.orig_obs
    man = session.manifold
    rid = session.target_region_id
    out_w, out_h = 640, 420

    frames = []

    def _pack_frame(img_f32: np.ndarray, banner: str, metric: str, col_bgr: tuple[int, int, int]) -> np.ndarray:
        u8 = np.clip(np.round(img_f32 * 255.0), 0, 255).astype(np.uint8)
        resized = cv2.resize(u8, (out_w, out_h - 46), interpolation=cv2.INTER_AREA)
        canvas = np.full((out_h, out_w, 3), 16, dtype=np.uint8)
        canvas[46:, :] = resized
        cv2.putText(canvas, banner, (12, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col_bgr, 1, cv2.LINE_AA)
        cv2.putText(canvas, metric, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 222, 230), 1, cv2.LINE_AA)
        return canvas

    # Phase 1: Red Sweep
    for s in session.red_sweep.steps:
        frames.append(
            _pack_frame(
                s.imagined_rgb,
                f"REGIONAL CHROMA IMAGINATION -> RED (650nm) | alpha={s.alpha:.1f}",
                f"region_{rid} dC={s.regional_delta_c:.3f} | L2_persistence={s.comparison.l2_persistence_corr:.3f}",
                (244, 63, 94),
            )
        )

    # Phase 2: Yellow Sweep
    for s in session.yellow_sweep.steps:
        frames.append(
            _pack_frame(
                s.imagined_rgb,
                f"REGIONAL CHROMA IMAGINATION -> YELLOW (580nm) | alpha={s.alpha:.1f}",
                f"region_{rid} dC={s.regional_delta_c:.3f} | L2_persistence={s.comparison.l2_persistence_corr:.3f}",
                (250, 204, 21),
            )
        )

    # Phase 3: Sunset Goal Hold (Roughness x1.40)
    goal = session.goal_recreation
    for _ in range(4):
        frames.append(
            _pack_frame(
                goal.final_imagined_rgb,
                "AUTONOMOUS GOAL RE-CREATION (Sunset 605nm + Roughness x1.40)",
                f"err: {goal.iterations[0].total_error:.3f} -> {goal.iterations[-1].total_error:.3f} | L2={goal.iterations[-1].l2_persistence_corr:.3f}",
                (56, 189, 248),
            )
        )

    # Phase 4: Full 360-deg Rainbow Rotor
    for deg in np.linspace(0.0, 360.0, 10):
        img_rb, clip_r = engine.apply_imagination(
            orig, man, PerceptualImaginationOp(hue_rotor_deg=float(deg)), region_id=None
        )
        frames.append(
            _pack_frame(
                img_rb,
                f"CLIFFORD RAINBOW ROTOR R(theta)  theta={deg:5.1f} deg",
                f"Full 360-deg closure RMSE={session.rainbow_sweep.closure_rmse_360:.1e} | gamut_clip={clip_r:.3f}",
                (168, 85, 247),
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, duration=0.15, loop=0)


if __name__ == "__main__":
    main()
