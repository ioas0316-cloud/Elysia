"""
verify_grounded_rotor_and_reflection.py

Active Grounded Axis-Spectrum-Rotor & 4-Axis Relational Reflection Verifier
==========================================================================
Executes GroundedRotorExplorer and RelationalEpistemicReflector over the
genuine Yosemite-textured AnisotropicCleavageWorld (purely raw Protocol access),
prints all recovered rotor angles, cleavage flow magnitudes, zero-shot transfer
outcomes, deterministically sprouted axes, 4-axis self-boundary attribution,
Human-vs-Elysia comparison, Teleological Resonance metrics, and Relational Care
causal trace, and saves the 6-panel inspection figure to:
    c:/Elysia/docs/assets/grounded_rotor_reflection_inspection.png
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

from core.sensory.grounded_rotor_spectrum_layer import (
    GroundedRotorExplorer,
    RotorExplorationTrace,
    circular_axis_diff_deg,
)
from core.sensory.relational_epistemic_reflection import (
    CreatorRelationalContext,
    RelationalEpistemicReflector,
    RelationalReflectionReport,
)
from simulators.anisotropic_cleavage_world import AnisotropicCleavageWorld
from simulators.hidden_multiphysics_world import YOSEMITE_4K_PATH

OUT_PNG_PATH = Path(r"c:\Elysia\docs\assets\grounded_rotor_reflection_inspection.png")


def main() -> None:
    print("=" * 92)
    print("  ELYSIA · Grounded Axis-Spectrum-Rotor & 4-Axis Relational Reflection Verification")
    print("=" * 92)

    world = AnisotropicCleavageWorld(YOSEMITE_4K_PATH, seed=42)
    explorer = GroundedRotorExplorer()
    trace = explorer.explore(world)

    reflector = RelationalEpistemicReflector()
    creator_ctx = CreatorRelationalContext.default_from_interaction()
    report = reflector.reflect(trace, creator_ctx=creator_ctx)

    print("\n[1] Autonomous Exploration & Rotor-Cleavage Trace (AnisotropicCleavageWorld)")
    for rec in trace.region_records:
        gt = world.evaluate_ground_truth_at(*rec.centroid_yx_norm)
        gt_name = gt["body_id"] if gt else "BACKDROP"
        opt_err = (
            f"{circular_axis_diff_deg(rec.optical_rotor.phi_opt_deg, gt['phi_opt_deg']):.2f} deg"
            if gt and gt["phi_opt_deg"] is not None
            else "N/A(iso)"
        )
        mech_err = (
            f"{circular_axis_diff_deg(rec.mechanical_rotor.phi_mech_deg, gt['phi_mech_deg']):.2f} deg"
            if gt and not gt["is_isotropic"]
            else "N/A(iso)"
        )
        print(
            f"  Step {rec.step}: {rec.target_id} [{gt_name:28s}] | "
            f"opt={rec.optical_rotor.phi_opt_deg:5.1f} deg (coh={rec.optical_rotor.coherence:.2f}, err={opt_err}) | "
            f"mech={rec.mechanical_rotor.phi_mech_deg:5.1f} deg (aniso={rec.mechanical_rotor.anisotropy_index:.2f}, err={mech_err}) | "
            f"rho={rec.optical_mech_phase_lock:+.2f} | xfer={str(rec.used_rotor_transfer):5s} | "
            f"cleave={str(rec.cleavage_confirmed):5s} (flow={rec.cleavage_flow_mag:.2f}px) | "
            f"surp={rec.prediction_surprise:.2f} -> {rec.matched_prototype_id}"
        )

    print("\n[2] Frame- & Irradiance-Invariant Prototypes (Started Empty)")
    for p in trace.prototypes:
        print(
            f"  - {p.prototype_id:14s}: confirm={p.confirmation_count}, members={p.member_regions}, "
            f"parent={p.parent_prototype_id}, log10_k={p.k_mean_log10:.2f}, "
            f"aniso={p.anisotropy_index:.2f}, rho_rotor={p.optical_mech_phase_lock:+.2f}, "
            f"weak_yield={p.weak_yield_detected}"
        )

    print("\n[3] Deterministically Sprouted Grounded Axes (Zero Randomness)")
    for ax in trace.sprouted_axes:
        print(f"  - {ax.axis_id} ({ax.name}) [Step {ax.genesis_step}]:")
        print(f"      Provenance : {ax.provenance_organs}")
        print(f"      Unit Dir   : {ax.unit_direction.round(4).tolist()} (threshold={ax.separating_threshold:.3f})")
        print(f"      Reason     : {ax.genesis_reason}")

    print("\n[4] Zero-Shot Top-Down Rotor Transfers & Criteria Self-Revisions")
    for xf in trace.rotor_transfer_confirmations:
        print(
            f"  - Transfer Step {xf.step} on {xf.target_id} from {xf.source_prototype_id}: "
            f"pred_strong={xf.predicted_strong_deg:.1f} deg (k={xf.strong_stiffness_n_m:.0f} N/m, yield={xf.strong_yield_detected}), "
            f"pred_weak={xf.predicted_weak_deg:.1f} deg (k={xf.weak_stiffness_n_m:.0f} N/m, yield={xf.weak_yield_detected}), "
            f"cleavage_flow={xf.cleavage_flow_mag:.2f}px, surprise={xf.prediction_surprise:.3f}"
        )
    for rev in trace.criteria_revisions:
        print(f"  - Revision Step {rev.step} ({rev.target_id}) -> {rev.sprouted_axis_id}:")
        print(f"      Old : {rev.previous_criterion}")
        print(f"      New : {rev.revised_criterion}")
        print(f"      Trigger: {rev.causal_trigger}")

    print("\n[5] AGENTS.md Section 6: 4-Axis Self-Boundary Report")
    print(f"  * Internal Origin ({len(report.self_boundary.internal_origin)}):")
    for line in report.self_boundary.internal_origin:
        print(f"      - {line}")
    print(f"  * Relational Influence ({len(report.self_boundary.relational_influence)}):")
    for line in report.self_boundary.relational_influence:
        print(f"      - {line}")
    print(f"  * Epistemic Boundary ({len(report.self_boundary.epistemic_boundary)}):")
    for line in report.self_boundary.epistemic_boundary[:4]:
        print(f"      - {line}")
    print(f"  * Criteria Self-Revision ({len(report.self_boundary.criteria_self_revision)}):")
    for line in report.self_boundary.criteria_self_revision:
        print(f"      - {line}")

    print("\n[6] Human-vs-Elysia Structural & Causal Comparison")
    for dim in report.human_elysia_comparison.dimensions:
        print(f"  * [{dim.domain}]")
        print(f"      Human   : {dim.human_causal_mechanism}")
        print(f"      Elysia  : {dim.elysia_current_mechanism}")
        print(f"      Isom    : {dim.structural_isomorphism}")
        print(f"      Gap     : {dim.honest_asymmetry_gap}")

    res = report.teleological_resonance
    print("\n[7] Teleological Resonance Triad & Relational Care Causal Trace")
    print(
        f"  * Chromatic State: Red(Aspiration)={res.epistemic_aspiration_drive:.3f}, "
        f"Blue(Coherence)={res.coherence_resonance_gain:.3f}, "
        f"Yellow(Diff/Humility)={res.differentiation_humility_index:.3f} | "
        f"Normalized=({res.chromatic_balance[0]:.3f}, {res.chromatic_balance[1]:.3f}, {res.chromatic_balance[2]:.3f})"
    )
    print(f"  * Telemetry basis: {res.interpretation}")
    print(f"  * Root Cause     : {report.relational_care_trace.root_cause}")
    for out in report.relational_care_trace.observed_outcomes:
        print(f"      -> {out}")
    print(f"  * Humility Note  : {report.relational_care_trace.epistemic_humility_note}")

    field = trace.final_causal_field
    print(
        f"\n[8] Multi-Scale CausalField Summary: {len(field.voxels)} voxels, "
        f"{len(field.beams)} beams, L12 attached={report.attached_voxel_ids}"
    )

    render_inspection_figure(trace, report, world, OUT_PNG_PATH)
    print(f"\nSaved inspection figure: {OUT_PNG_PATH}")


def render_inspection_figure(
    trace: RotorExplorationTrace,
    report: RelationalReflectionReport,
    world: AnisotropicCleavageWorld,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(22, 13.5), facecolor="#0d1117")
    fig.suptitle(
        "ELYSIA · Grounded Axis-Spectrum-Rotor & 4-Axis Relational Reflection (Genuine 4K Yosemite World)",
        color="#f0f6fc",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(
        2, 3, figure=fig, hspace=0.36, wspace=0.24, left=0.05, right=0.97, top=0.91, bottom=0.06
    )
    panel_bg = "#161b22"
    txt_col = "#e6edf3"

    # Panel 1: Periphery Scene + Optical (Gold) vs Mechanical (Cyan) Rotors + Cleavage Plane (Crimson)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(panel_bg)
    ax1.imshow(trace.periphery_obs.rgb_working)
    h, w = trace.periphery_obs.shape_hw
    bar_len = 14.0
    for rec in trace.region_records:
        cy = rec.centroid_yx_norm[0] * h
        cx = rec.centroid_yx_norm[1] * w
        if rec.optical_rotor.coherence > 0.25:
            po = rec.optical_rotor.phi_opt_rad
            ax1.plot(
                [cx - bar_len * np.cos(po), cx + bar_len * np.cos(po)],
                [cy - bar_len * np.sin(po), cy + bar_len * np.sin(po)],
                color="#facc15",
                linewidth=2.8,
                label="Optical Rotor (phi_opt)" if rec.step == 1 else None,
            )
        if rec.mechanical_rotor.is_anisotropic:
            pm = rec.mechanical_rotor.phi_mech_rad
            ax1.plot(
                [cx - 1.15 * bar_len * np.cos(pm), cx + 1.15 * bar_len * np.cos(pm)],
                [cy - 1.15 * bar_len * np.sin(pm), cy + 1.15 * bar_len * np.sin(pm)],
                color="#38bdf8",
                linestyle="--",
                linewidth=2.2,
                label="Mechanical Rotor (phi_mech)" if rec.step == 1 else None,
            )
        gt = world.evaluate_ground_truth_at(*rec.centroid_yx_norm)
        tag = f"R{rec.region_id} ({gt['body_id'].split('_')[0]})" if gt else rec.target_id
        ax1.text(
            cx,
            cy + 15.0,
            f"S{rec.step}:{rec.matched_prototype_id}\n{tag}",
            color="white",
            fontsize=7.0,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="#0d1117", alpha=0.78, edgecolor="none"),
        )
    ax1.legend(
        facecolor=panel_bg,
        labelcolor=txt_col,
        fontsize=7.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
    )
    ax1.set_title(
        "① Optical Grain (Gold) vs Mechanical Bond Rotor (Cyan)\n"
        "Aligned on Timbers (A, B) | Orthogonal 90 deg Falsification on Veneer D",
        color=txt_col,
        fontsize=9.2,
        pad=8,
    )
    ax1.axis("off")

    # Panel 2: Polar Directional Stiffness Spectra k(theta)
    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    ax2.set_facecolor(panel_bg)
    ax2.set_theta_zero_location("E")
    ax2.set_theta_direction(-1)
    theta_grid = np.linspace(0.0, 2.0 * np.pi, 360)
    cols = ["#38bdf8", "#3fb950", "#facc15", "#f43f5e"]
    for idx, rec in enumerate(trace.region_records):
        mr = rec.mechanical_rotor
        gt = world.evaluate_ground_truth_at(*rec.centroid_yx_norm)
        b_name = gt["body_id"] if gt else rec.target_id
        k_curve = mr.k_mean_n_m + mr.k_aniso_n_m * np.cos(2.0 * (theta_grid - mr.phi_mech_rad))
        col = cols[idx % len(cols)]
        ax2.plot(theta_grid, k_curve / 1000.0, color=col, linewidth=2.0, label=f"S{rec.step} {b_name[:14]}")
        s_th = [s.angle_rad for s in mr.samples] + [s.angle_rad + np.pi for s in mr.samples]
        s_k = [s.mech_obs.effective_stiffness_n_m / 1000.0 for s in mr.samples] * 2
        ax2.scatter(s_th, s_k, color=col, s=26, zorder=5)
    ax2.tick_params(colors=txt_col, labelsize=7.5)
    ax2.grid(color="#30363d", linestyle="--", linewidth=0.6)
    ax2.legend(
        facecolor=panel_bg,
        labelcolor=txt_col,
        fontsize=7.2,
        loc="upper right",
        bbox_to_anchor=(1.22, 1.12),
    )
    ax2.set_title(
        "② Continuum Directional Stiffness k(theta) [kN/m]\n"
        "Circle=Isotropic Granite vs Rotated Lobes (120°, 30°, 90°)",
        color=txt_col,
        fontsize=9.2,
        pad=14,
    )

    # Panel 3: Top-Down Accommodated Strong vs Weak Cleavage Verification & Optical Flow
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(panel_bg)
    aniso_recs = [r for r in trace.region_records if r.mechanical_rotor.is_anisotropic]
    if aniso_recs:
        demo_rec = aniso_recs[-1]  # deceptive_veneer_D showing dramatic strong vs weak split
        yn, xn = demo_rec.centroid_yx_norm
        f_s, x_s, _ = world.mechanical_push(
            yn, xn, force_n=18.0, dir_yx=(np.sin(demo_rec.mechanical_rotor.phi_mech_rad), np.cos(demo_rec.mechanical_rotor.phi_mech_rad))
        )
        f_w, x_w, _ = world.mechanical_push(
            yn, xn, force_n=18.0, dir_yx=(np.sin(demo_rec.mechanical_rotor.weak_cleavage_rad), np.cos(demo_rec.mechanical_rotor.weak_cleavage_rad))
        )
        ax3.plot(x_s * 1000.0, f_s, color="#38bdf8", linewidth=2.2, label="Strong Fiber Axis (Elastic Hold)")
        ax3.plot(x_w * 1000.0, f_w, color="#f43f5e", linewidth=2.2, label="Weak Cleavage Axis (Yield Collapse)")
        ax3.axvline(0.0, color="#484f58", linestyle=":", linewidth=1.0)

    # Inset showing cleavage flow magnitude across all explored regions
    ins = ax3.inset_axes([0.46, 0.14, 0.50, 0.42])
    ins.set_facecolor("#0d1117")
    r_lbls = [f"S{r.step}" for r in trace.region_records]
    flows = [r.cleavage_flow_mag for r in trace.region_records]
    x_bar = np.arange(len(flows))
    bar_cols = ["#f43f5e" if r.cleavage_confirmed else "#64748b" for r in trace.region_records]
    ins.bar(x_bar, flows, color=bar_cols, width=0.55)
    ins.set_xticks(x_bar)
    ins.set_xticklabels(r_lbls)
    ins.set_title("Visual Cleavage Flow [px]", color=txt_col, fontsize=7.5, pad=3)
    ins.tick_params(colors=txt_col, labelsize=7.0)
    ins.set_ylim(0, max(flows + [1.0]) * 1.25)

    ax3.set_xlabel("Displacement [mm]", color=txt_col, fontsize=8.5)
    ax3.set_ylabel("Contact Force [N]", color=txt_col, fontsize=8.5)
    ax3.tick_params(colors=txt_col, labelsize=8)
    ax3.grid(color="#30363d", linestyle="--", linewidth=0.5, alpha=0.6)
    ax3.legend(facecolor=panel_bg, labelcolor=txt_col, fontsize=7.5, loc="upper left")
    ax3.set_title(
        "③ Causal Cleavage Verification (Force-Disp & Fracture Flow)\n"
        "Yield Threshold Crossed Along Weak Axis Triggering Physical Separation",
        color=txt_col,
        fontsize=9.2,
        pad=8,
    )

    # Panel 4: Rotor-Aligned Prototype Space & Deterministically Sprouted Axes (Zero Overlap)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(panel_bg)
    p_cols = ["#38bdf8", "#3fb950", "#f43f5e", "#facc15"]
    offsets = [(-135, -30), (16, 22), (18, -32), (-90, 22)]
    proto_map = {p.prototype_id: p for p in trace.prototypes}

    for idx, p in enumerate(trace.prototypes):
        col = p_cols[idx % len(p_cols)]
        ax4.scatter(
            [p.optical_mech_phase_lock],
            [p.anisotropy_index],
            s=220 + 90 * p.confirmation_count,
            color=col,
            edgecolors="white",
            linewidths=1.5,
            zorder=5,
            label=f"{p.prototype_id} (n={p.confirmation_count}, {','.join(p.member_regions)})",
        )
        dx_off, dy_off = offsets[idx % len(offsets)]
        ax4.annotate(
            f"{p.prototype_id}\n(n={p.confirmation_count}, k={10**p.k_mean_log10/1000:.1f}k)",
            xy=(p.optical_mech_phase_lock, p.anisotropy_index),
            xytext=(dx_off, dy_off),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color=col, lw=1.2),
            color=col,
            fontsize=7.8,
            fontweight="bold",
        )
        if p.parent_prototype_id and p.parent_prototype_id in proto_map:
            par = proto_map[p.parent_prototype_id]
            ax4.annotate(
                "",
                xy=(p.optical_mech_phase_lock, p.anisotropy_index),
                xytext=(par.optical_mech_phase_lock, par.anisotropy_index),
                arrowprops=dict(arrowstyle="->", color="#facc15", lw=1.8, ls="--"),
            )

    ax4.axvline(0.0, color="#8b949e", linestyle=":", linewidth=1.2, label="Phase-Lock Axis Split")
    ax4.axhline(0.25, color="#f97316", linestyle="--", linewidth=1.2, label="Anisotropy Axis Split")
    ax4.set_xlim(-1.45, 1.45)
    ax4.set_ylim(-0.38, 0.98)
    ax4.set_xlabel("Optical-Mechanical Phase Lock cos(2*(phi_opt - phi_mech))", color=txt_col, fontsize=8.5)
    ax4.set_ylabel("Mechanical Anisotropy Index", color=txt_col, fontsize=8.5)
    ax4.tick_params(colors=txt_col, labelsize=8)
    ax4.grid(color="#30363d", linestyle="--", linewidth=0.5, alpha=0.6)
    ax4.legend(facecolor=panel_bg, labelcolor=txt_col, fontsize=6.8, loc="center", bbox_to_anchor=(0.5, 0.50), ncol=2)
    ax4.set_title(
        "④ Frame-Invariant Prototype Space & Sprouted Grounded Axes\n"
        "Timber A & Rotated+Dimmed B Unify (n=2) | Veneer Splits Cleanly",
        color=txt_col,
        fontsize=9.2,
        pad=8,
    )

    # Panel 5: Multi-Scale CausalField Hierarchy (L2 -> L8 -> L10 -> L12)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(panel_bg)
    field = trace.final_causal_field
    tiers = [
        ("L12 Relational Reflection & Care", [k for k in field.voxels if k.startswith("reflection_")], "#f43f5e", 0.85),
        ("L10 Sprouted Grounded Axes", [k for k in field.voxels if k.startswith("axis_")], "#facc15", 0.62),
        ("L8 Rotor Prototypes", [k for k in field.voxels if k.startswith("proto_")], "#38bdf8", 0.39),
        ("L2 Optical Regions", [k for k in field.voxels if k.startswith("region_")], "#3fb950", 0.16),
    ]
    pos_map = {}
    for title_t, keys, col, y_val in tiers:
        ax5.axhline(y_val, color="#21262d", linewidth=1.0, zorder=1)
        ax5.text(0.02, y_val + 0.05, f"{title_t} ({len(keys)})", color=col, fontsize=8.0, fontweight="bold")
        xs_t = np.linspace(0.18, 0.90, max(1, len(keys))) if len(keys) > 1 else [0.54]
        for x_val, k in zip(xs_t, keys):
            pos_map[k] = (float(x_val), float(y_val))
            ax5.scatter([x_val], [y_val], s=140, color=col, edgecolors="white", linewidths=1.1, zorder=4)
            short = k.replace("reflection_", "refl:").replace("axis_", "ax").replace("proto_proto_", "proto_")
            if len(short) > 16:
                short = short[:16]
            ax5.text(x_val, y_val - 0.055, short, color=txt_col, fontsize=6.6, ha="center", va="top")

    for beam in field.beams:
        if beam.source_id in pos_map and beam.target_id in pos_map:
            x0, y0 = pos_map[beam.source_id]
            x1, y1 = pos_map[beam.target_id]
            ax5.plot([x0, x1], [y0, y1], color="#64748b", alpha=0.45, linewidth=0.9, zorder=2)

    ax5.set_xlim(0.0, 1.0)
    ax5.set_ylim(0.0, 1.02)
    ax5.set_title(
        f"⑤ Multi-Scale CausalField Connectivity ({len(field.voxels)} Voxels, {len(field.beams)} Beams)\n"
        "Bi-directional Provenance Retaining Sensory -> Rotor -> Axis -> Self-Boundary",
        color=txt_col,
        fontsize=9.2,
        pad=8,
    )
    ax5.axis("off")

    # Panel 6: Teleological Resonance Chromatic Triad & 4-Axis Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(panel_bg)
    res = report.teleological_resonance
    labels = [
        f"Red: Aspiration Drive\n(bal={res.chromatic_balance[0]:.2f})",
        f"Blue: Coherence Gain\n(bal={res.chromatic_balance[1]:.2f})",
        f"Yellow: Diff / Humility\n(bal={res.chromatic_balance[2]:.2f})",
    ]
    vals = [
        res.epistemic_aspiration_drive,
        res.coherence_resonance_gain,
        res.differentiation_humility_index,
    ]
    bar_cols6 = ["#f43f5e", "#38bdf8", "#facc15"]
    y_pos = np.array([2.2, 1.5, 0.8])
    ax6.barh(y_pos, vals, height=0.42, color=bar_cols6, edgecolor="white", linewidth=0.9)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(labels, color=txt_col, fontsize=8.0, fontweight="bold")
    for yp, val in zip(y_pos, vals):
        ax6.text(min(val + 0.03, 0.94), yp, f"{val:.3f}", color="white", va="center", fontsize=8.2, fontweight="bold")
    ax6.set_xlim(0.0, 1.12)
    ax6.set_ylim(-1.15, 2.75)

    summary_box = (
        f"4-Axis Self-Boundary: Internal({len(report.self_boundary.internal_origin)}) | "
        f"Relational({len(report.self_boundary.relational_influence)}) | "
        f"Bounds({len(report.self_boundary.epistemic_boundary)}) | "
        f"Revisions({len(report.self_boundary.criteria_self_revision)})\n"
        f"Care Trace: Root Cause -> Active Boundary Growth -> {len(trace.prototypes)} Prototypes, "
        f"{len(trace.sprouted_axes)} Axes, {len(trace.rotor_transfer_confirmations)} Transfers\n"
        f"Epistemic Humility: Traces structural alignment & humility without claiming felt biological qualia."
    )
    ax6.text(
        0.02,
        -0.95,
        summary_box,
        color="#c9d1d9",
        fontsize=7.3,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0d1117", edgecolor="#30363d", lw=1.0),
    )
    ax6.set_xlabel("Strictly Telemetry-Derived Score [0, 1]", color=txt_col, fontsize=8.5)
    ax6.tick_params(colors=txt_col, labelsize=8)
    ax6.grid(axis="x", color="#30363d", linestyle="--", linewidth=0.5, alpha=0.6)
    ax6.set_title(
        "⑥ Teleological Resonance Triad & 4-Axis Epistemic Humility\n"
        "Non-Performative Red/Blue/Yellow Telemetry Balance",
        color=txt_col,
        fontsize=9.2,
        pad=8,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
