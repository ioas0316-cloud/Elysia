from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.sensory.spacetime_causal_game_video_engine import (
    ContinuousCausalArena,
    GameSpacetimeReport,
    RealVideoSpacetimeReport,
    SpacetimeCausalDiscoveryEngine,
)

VIDEO_PATH = Path(r"c:\Elysia\demos\real_world_assets\real_external_nature_video.mp4")
STATE_PATH = Path(r"c:\Elysia\data\spacetime_causal_game_video_state.json")
DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_spacetime_causal_game_video_dashboard.png"
)


def render_dashboard(
    game_rep: GameSpacetimeReport,
    video_rep: RealVideoSpacetimeReport,
    arena_final_rgb: np.ndarray,
    engine: SpacetimeCausalDiscoveryEngine,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), dpi=150)
    fig.patch.set_facecolor("#F8FAFC")

    disc_map = {d.entity_role: d for d in game_rep.phase1_discoveries}

    # Panel 1: Live Arena RGB + Blindly Discovered Entities
    ax1 = axes[0, 0]
    ax1.imshow(game_rep.sample_frame_rgb)
    d_self = disc_map["SELF_BODY"]
    d_ext = disc_map["EXTERNAL_AUTONOMOUS_AGENT"]
    d_sh = disc_map["ILLUMINATION_SHADOW"]
    ty, tx = game_rep.discovered_target_yx

    ax1.scatter(
        [d_self.estimated_yx_px[1]],
        [d_self.estimated_yx_px[0]],
        s=150,
        facecolors="none",
        edgecolors="#22D3EE",
        lw=2.5,
        label="SELF",
        zorder=5,
    )
    ax1.scatter(
        [d_ext.estimated_yx_px[1]],
        [d_ext.estimated_yx_px[0]],
        s=120,
        c="#FB923C",
        marker="^",
        edgecolors="#0F172A",
        lw=1.2,
        label="EXT AGENT",
        zorder=5,
    )
    ax1.scatter(
        [d_sh.estimated_yx_px[1]],
        [d_sh.estimated_yx_px[0]],
        s=110,
        c="#E879F9",
        marker="X",
        edgecolors="#0F172A",
        lw=1.0,
        label="SHADOW",
        zorder=5,
    )
    ax1.scatter(
        [tx],
        [ty],
        s=150,
        c="#FACC15",
        marker="*",
        edgecolors="#0F172A",
        lw=1.0,
        label="TARGET",
        zorder=5,
    )
    ax1.legend(loc="lower right", fontsize=7.5, framealpha=0.88, ncol=2)
    ax1.set_title(
        f"1. Blind Arena Entity Discovery (Pixels Only)\n"
        f"(SelfErr={game_rep.self_localization_error_px:.1f}px | "
        f"ExtErr={game_rep.external_agent_localization_error_px:.1f}px)",
        fontsize=9.8,
        fontweight="bold",
        pad=8,
    )
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Panel 2: Motor-Efference Causal Resonance Heatmap
    ax2 = axes[0, 1]
    ax2.imshow(game_rep.self_resonance_heatmap, cmap="inferno")
    ax2.scatter(
        [d_self.estimated_yx_px[1]],
        [d_self.estimated_yx_px[0]],
        s=130,
        facecolors="none",
        edgecolors="#22D3EE",
        lw=2.0,
    )
    ax2.set_title(
        f"2. Motor-Flow Resonance Map (Isolates SELF)\n"
        f"(Learned Gain y={game_rep.learned_motor_gain_yx[0]:.2f}, "
        f"x={game_rep.learned_motor_gain_yx[1]:.2f} px/cmd)",
        fontsize=9.8,
        fontweight="bold",
        pad=8,
    )
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Panel 3: Closed-Loop Blind Visual-Motor Control Trajectory
    ax3 = axes[0, 2]
    ax3.imshow(arena_final_rgb)
    est_ys = [p[0] for p in game_rep.control_trajectory_est_yx]
    est_xs = [p[1] for p in game_rep.control_trajectory_est_yx]
    true_ys = [p[0] for p in game_rep.control_trajectory_true_yx]
    true_xs = [p[1] for p in game_rep.control_trajectory_true_yx]
    ax3.plot(true_xs, true_ys, color="#FACC15", lw=2.2, ls="--", label="True SELF Path")
    ax3.plot(est_xs, est_ys, color="#22D3EE", lw=2.0, label="Blind Visual Track")
    ax3.scatter([est_xs[0]], [est_ys[0]], s=80, c="#38BDF8", edgecolors="#FFFFFF", zorder=5)
    ax3.scatter([est_xs[-1]], [est_ys[-1]], s=110, c="#22C55E", edgecolors="#FFFFFF", zorder=6)
    ax3.legend(loc="lower right", fontsize=8.0, framealpha=0.88)
    ax3.set_title(
        f"3. Closed-Loop Visual-Motor Control\n"
        f"(Dist {game_rep.control_initial_dist_px:.0f}px -> {game_rep.control_final_dist_px:.0f}px | "
        f"-{game_rep.control_distance_reduction_ratio * 100:.0f}%)",
        fontsize=9.8,
        fontweight="bold",
        pad=8,
    )
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Panel 4: Real External Nature Video Autonomous Agent Discovery
    ax4 = axes[1, 0]
    vh, vw = video_rep.sample_frame_rgb.shape[:2]
    disp_w = 480
    disp_h = max(64, int(round(vh * (disp_w / float(vw)))))
    v_disp = cv2.resize(video_rep.sample_frame_rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    f_disp = cv2.resize(video_rep.sample_flow_mag, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
    ax4.imshow(v_disp)
    ax4.imshow(f_disp, cmap="inferno", alpha=0.40)
    sy = disp_h / float(vh)
    sx = disp_w / float(vw)
    v_ys = [p[0] * sy for p in video_rep.discovered_agent_trajectory_yx]
    v_xs = [p[1] * sx for p in video_rep.discovered_agent_trajectory_yx]
    if len(v_xs) >= 2:
        ax4.plot(v_xs, v_ys, color="#38BDF8", lw=2.2)
    ax4.scatter([v_xs[-1]], [v_ys[-1]], s=90, c="#22C55E", edgecolors="#0F172A", lw=1.2, zorder=6)
    ax4.set_title(
        f"4. Real Nature Video Agent Discovery\n"
        f"(Inferred Autonomous Agency={video_rep.inferred_autonomous_agency_score:.2f} from Pixels)",
        fontsize=9.6,
        fontweight="bold",
        pad=8,
    )
    ax4.set_xticks([])
    ax4.set_yticks([])

    # Panel 5: Distance-to-Target Convergence Curve
    ax5 = axes[1, 1]
    ax5.plot(
        range(len(game_rep.distance_curve_px)),
        game_rep.distance_curve_px,
        color="#0284C7",
        marker="o",
        lw=2.2,
    )
    ax5.set_xlabel("Control Step (Closed-Loop RGB -> Action)", fontsize=8.8)
    ax5.set_ylabel("True Distance to Target (px)", fontsize=8.8)
    ax5.grid(True, alpha=0.25)
    ax5.set_ylim(0.0, max(game_rep.distance_curve_px) * 1.15 + 5.0)
    ax5.set_title(
        "5. Visual-Motor Control Convergence\n(Zero Game State / Coordinate Peeking)",
        fontsize=9.8,
        fontweight="bold",
        pad=8,
    )

    # Panel 6: Summary Metrics & CausalField Telemetry
    ax6 = axes[1, 2]
    m_names = ["Self Loc\nAccuracy", "Ext Agent\nAccuracy", "Dist Reduc\nRatio", "Video Inf\nAgency"]
    m_vals = [
        max(0.0, 1.0 - game_rep.self_localization_error_px / 40.0),
        max(0.0, 1.0 - game_rep.external_agent_localization_error_px / 40.0),
        game_rep.control_distance_reduction_ratio,
        video_rep.inferred_autonomous_agency_score,
    ]
    cols = ["#0284C7", "#EA580C", "#16A34A", "#7C3AED"]
    bars = ax6.bar(np.arange(len(m_names)), m_vals, color=cols, width=0.55)
    for b, val in zip(bars, m_vals):
        ax6.text(
            b.get_x() + b.get_width() / 2,
            val + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#0F172A",
        )
    ax6.set_xticks(np.arange(len(m_names)))
    ax6.set_xticklabels(m_names, fontsize=8.5, fontweight="bold")
    ax6.set_ylim(0.0, 1.22)
    ax6.grid(axis="y", alpha=0.25)
    ax6.set_title(
        f"6. Spacetime Discovery & Control Scores\n"
        f"CausalField: {len(engine.causal_field.voxels)} Voxels, "
        f"{len(engine.causal_field.beams)} Beams",
        fontsize=9.8,
        fontweight="bold",
        pad=8,
    )

    fig.suptitle(
        "ELYSIA Spacetime Causal Flow Discovery & Closed-Loop Control (Headless Pygame + Real Video, Zero VLM/LLM)",
        fontsize=13.0,
        fontweight="bold",
        y=0.982,
    )
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.945], h_pad=2.4)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return output_path


def run_demo() -> Dict[str, Any]:
    t0 = time.perf_counter()
    engine = SpacetimeCausalDiscoveryEngine()
    arena = ContinuousCausalArena(width=240, height=180)

    game_rep = engine.discover_and_control_arena(arena, control_steps=22)
    final_arena_rgb = arena.render()

    video_rep = engine.discover_autonomous_agent_in_real_video(
        VIDEO_PATH, max_frames=10, frame_stride=6, max_width=240
    )

    dash_path = render_dashboard(
        game_rep=game_rep,
        video_rep=video_rep,
        arena_final_rgb=final_arena_rgb,
        engine=engine,
        output_path=DASHBOARD_PATH,
    )

    elapsed_sec = time.perf_counter() - t0
    state = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed_sec, 2),
        "vlm_used": False,
        "llm_used": False,
        "hidden_game_state_used_by_agent": False,
        "phase1_discoveries": [
            {
                "entity_role": d.entity_role,
                "estimated_yx_px": [round(d.estimated_yx_px[0], 2), round(d.estimated_yx_px[1], 2)],
                "mean_velocity_yx": [round(d.mean_velocity_yx[0], 3), round(d.mean_velocity_yx[1], 3)],
                "motor_causal_correlation": round(d.motor_causal_correlation, 3),
                "autonomous_motion_score": round(d.autonomous_motion_score, 3),
                "voxel_id": d.voxel_id,
            }
            for d in game_rep.phase1_discoveries
        ],
        "game_metrics": {
            "self_localization_error_px": round(game_rep.self_localization_error_px, 2),
            "external_agent_localization_error_px": round(
                game_rep.external_agent_localization_error_px, 2
            ),
            "learned_motor_gain_yx": [
                round(game_rep.learned_motor_gain_yx[0], 3),
                round(game_rep.learned_motor_gain_yx[1], 3),
            ],
            "discovered_target_yx": [
                round(game_rep.discovered_target_yx[0], 2),
                round(game_rep.discovered_target_yx[1], 2),
            ],
            "control_initial_dist_px": round(game_rep.control_initial_dist_px, 2),
            "control_final_dist_px": round(game_rep.control_final_dist_px, 2),
            "control_distance_reduction_ratio": round(
                game_rep.control_distance_reduction_ratio, 4
            ),
        },
        "real_video_metrics": {
            "video_name": video_rep.video_name,
            "frames_processed": video_rep.frames_processed,
            "inferred_autonomous_agency_score": round(
                video_rep.inferred_autonomous_agency_score, 4
            ),
            "voxel_id": video_rep.voxel_id,
        },
        "causal_field_summary": {
            "voxel_count": len(engine.causal_field.voxels),
            "beam_count": len(engine.causal_field.beams),
        },
        "dashboard_path": str(dash_path),
    }

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 92)
    print("ELYSIA Spacetime Causal Flow Discovery & Closed-Loop Control (Zero VLM/LLM)")
    print("=" * 92)
    print(f"Elapsed Time          : {elapsed_sec:.2f}s")
    print(
        f"Phase 1 Blind Discovery: Self Err = {game_rep.self_localization_error_px:.2f}px | "
        f"External Agent Err = {game_rep.external_agent_localization_error_px:.2f}px | "
        f"Learned Gain = ({game_rep.learned_motor_gain_yx[0]:.2f}, {game_rep.learned_motor_gain_yx[1]:.2f}) px/cmd"
    )
    print(
        f"Phase 2 Blind Control  : Distance {game_rep.control_initial_dist_px:.1f}px -> "
        f"{game_rep.control_final_dist_px:.1f}px (-{game_rep.control_distance_reduction_ratio * 100:.1f}%)"
    )
    print(
        f"Phase 3 Real Video     : {video_rep.video_name}.mp4 ({video_rep.frames_processed} frames) -> "
        f"Bottom-up Inferred Autonomous Agency = {video_rep.inferred_autonomous_agency_score:.3f}"
    )
    print(
        f"CausalField (6D)       : {len(engine.causal_field.voxels)} voxels, "
        f"{len(engine.causal_field.beams)} connectivity beams"
    )
    print(f"State Saved            : {STATE_PATH}")
    print(f"Dashboard Saved        : {dash_path}")
    return state


if __name__ == "__main__":
    run_demo()
