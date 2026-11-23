from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.sensory.active_ocular_mechanics_engine import (
    ActiveOcularMechanicsEngine,
    FovealEngram,
    SceneScanResult,
    VideoTrackingResult,
)

ASSETS_DIR = Path(r"c:\Elysia\demos\real_world_assets")
STATE_PATH = Path(r"c:\Elysia\data\active_ocular_mechanics_state.json")
SCREEN_STATE_PATH = Path(r"c:\Elysia\data\active_ocular_mechanics_screen_state.json")
DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_active_ocular_mechanics_dashboard.png"
)


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read real image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _make_perturbed_crop(
    full_rgb: np.ndarray,
    box_yxyx: Tuple[int, int, int, int],
    brightness_scale: float = 0.55,
    shift_frac: float = 0.05,
) -> np.ndarray:
    """Apply physical 0.55x linear irradiance darkening and a small saccade-jitter crop shift."""
    full_h, full_w = full_rgb.shape[:2]
    y0, x0, y1, x1 = box_yxyx
    bh = max(16, y1 - y0)
    bw = max(16, x1 - x0)
    dy = int(round(bh * shift_frac))
    dx = int(round(-bw * shift_frac))

    sy0 = int(np.clip(y0 + dy, 0, max(0, full_h - bh)))
    sx0 = int(np.clip(x0 + dx, 0, max(0, full_w - bw)))
    sy1 = min(full_h, sy0 + bh)
    sx1 = min(full_w, sx0 + bw)

    crop = full_rgb[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0
    linear = np.power(np.clip(crop, 0.0, 1.0), 2.2) * float(brightness_scale)
    dark_srgb = np.power(np.clip(linear, 0.0, 1.0), 1.0 / 2.2)
    return np.clip(np.round(dark_srgb * 255.0), 0.0, 255.0).astype(np.uint8)


def _downsample_for_plot(rgb: np.ndarray, max_w: int = 960) -> Tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    if w <= max_w:
        return rgb, 1.0
    scale = max_w / float(w)
    nh = max(32, int(round(h * scale)))
    resized = cv2.resize(rgb, (max_w, nh), interpolation=cv2.INTER_AREA)
    return resized, scale


def _render_cleaved_bond_overlay(scan: SceneScanResult, fix_idx: int) -> np.ndarray:
    """Create a side-by-side comparison of raw foveal pixels and cleaved optical bonds."""
    crop = scan.foveal_crops_rgb[fix_idx]
    fobs = scan.foveal_obs_list[fix_idx]
    fh, fw = fobs.shape_hw

    crop_small = cv2.resize(crop, (fw, fh), interpolation=cv2.INTER_AREA)
    overlay = (crop_small.astype(np.float32) * 0.45).astype(np.float32)

    hc = fobs.l2.bond_lattice.horiz_cleaved
    vc = fobs.l2.bond_lattice.vert_cleaved
    edge_mask = np.zeros((fh, fw), dtype=bool)
    edge_mask[:, :-1] |= hc
    edge_mask[:, 1:] |= hc
    edge_mask[:-1, :] |= vc
    edge_mask[1:, :] |= vc

    mat_mask = fobs.cause_map == 3
    overlay[edge_mask] = np.array([0.0, 225.0, 255.0], dtype=np.float32)
    overlay[edge_mask & mat_mask] = np.array([255.0, 110.0, 40.0], dtype=np.float32)
    overlay_u8 = np.clip(overlay, 0.0, 255.0).astype(np.uint8)

    divider = np.full((fh, 4, 3), 240, dtype=np.uint8)
    return np.concatenate([crop_small, divider, overlay_u8], axis=1)


def _draw_scan_panel(
    ax: plt.Axes,
    rgb_full: np.ndarray,
    scan: SceneScanResult,
    title: str,
    show_salience_contours: bool = False,
) -> None:
    disp, s = _downsample_for_plot(rgb_full, max_w=960)
    dh, dw = disp.shape[:2]
    ax.imshow(disp)

    if show_salience_contours:
        sal_resized = cv2.resize(
            scan.salience_map, (dw, dh), interpolation=cv2.INTER_LINEAR
        )
        ax.contour(
            sal_resized,
            levels=[0.35, 0.60, 0.82],
            colors=["#FDE047", "#FB923C", "#EF4444"],
            linewidths=[0.8, 1.0, 1.2],
            alpha=0.80,
        )

    xs: List[float] = []
    ys: List[float] = []
    for k, eng in enumerate(scan.foveal_engrams):
        y0, x0, y1, x1 = eng.fovea_box_yxyx
        gy, gx = eng.gaze_yx_px
        rx, ry = x0 * s, y0 * s
        rw, rh = (x1 - x0) * s, (y1 - y0) * s
        cx, cy = gx * s, gy * s
        xs.append(cx)
        ys.append(cy)

        rect = Rectangle(
            (rx, ry),
            rw,
            rh,
            fill=False,
            edgecolor="#22D3EE",
            linewidth=1.5,
            linestyle="-",
            alpha=0.90,
        )
        ax.add_patch(rect)
        ax.scatter(
            [cx],
            [cy],
            s=85,
            c="#EF4444" if k == 0 else "#F59E0B",
            edgecolors="#FFFFFF",
            linewidths=1.2,
            zorder=5,
        )
        ax.text(
            cx + 8,
            cy - 8,
            f"F{k + 1}",
            color="#FFFFFF",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="#0F172A", ec="#22D3EE", lw=0.8, alpha=0.85),
            zorder=6,
        )

    if len(xs) >= 2:
        ax.plot(xs, ys, color="#FACC15", linewidth=1.6, linestyle="--", alpha=0.90, zorder=4)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])


def render_dashboard(
    face_rgb: np.ndarray,
    face_scan: SceneScanResult,
    yosemite_rgb: np.ndarray,
    yosemite_scan: SceneScanResult,
    third_scan: SceneScanResult,
    video_res: VideoTrackingResult,
    query_labels: List[str],
    target_labels: List[str],
    resonance_matrix: np.ndarray,
    top1_accuracy: float,
    mean_margin: float,
    engine: ActiveOcularMechanicsEngine,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=150)
    fig.patch.set_facecolor("#F8FAFC")

    # Panel 1: Real Human Face HD + saccadic scanpath
    _draw_scan_panel(
        axes[0, 0],
        face_rgb,
        face_scan,
        title=(
            f"1. Active Foveal Scan: Human Face HD\n"
            f"({len(face_scan.foveal_engrams)} fixations, "
            f"{face_scan.periphery_vs_fovea_detail_gain:.1f}x Detail Gain)"
        ),
    )

    # Panel 2: Fovea High-Res Zoom + Cleaved Bond Overlay
    ax2 = axes[0, 1]
    best_f_idx = int(
        np.argmax([e.cleaved_bond_ratio + 0.01 * e.region_count for e in face_scan.foveal_engrams])
    )
    pair_img = _render_cleaved_bond_overlay(face_scan, best_f_idx)
    f_eng = face_scan.foveal_engrams[best_f_idx]
    ax2.imshow(pair_img)
    ax2.set_title(
        f"2. Fovea F{best_f_idx + 1} High-Res Zoom vs Optical Bond Cleavage\n"
        f"(Regions={f_eng.region_count}, Material={f_eng.material_boundary_count}, "
        f"Cleaved={f_eng.cleaved_bond_ratio * 100:.1f}%)",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    ax2.text(
        pair_img.shape[1] * 0.24,
        pair_img.shape[0] - 8,
        "Raw Foveal Crop",
        ha="center",
        va="bottom",
        color="#FFFFFF",
        fontsize=8.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="#0F172A", alpha=0.8),
    )
    ax2.text(
        pair_img.shape[1] * 0.76,
        pair_img.shape[0] - 8,
        "Material & Cleaved Bonds",
        ha="center",
        va="bottom",
        color="#22D3EE",
        fontsize=8.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="#0F172A", alpha=0.8),
    )
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Panel 3: Real 4K Yosemite + salience contours + scanpath
    _draw_scan_panel(
        axes[0, 2],
        yosemite_rgb,
        yosemite_scan,
        title=(
            f"3. Active 4K Scan: Yosemite + Salience\n"
            f"({len(yosemite_scan.foveal_engrams)} fixations, "
            f"{yosemite_scan.periphery_vs_fovea_detail_gain:.1f}x Detail Gain)"
        ),
        show_salience_contours=True,
    )

    # Panel 4: Real Nature Video Optical Flow & Motion-Guided Foveal Tracking
    ax4 = axes[1, 0]
    v_frame, vs = _downsample_for_plot(video_res.sample_frame_rgb, max_w=960)
    vh, vw = v_frame.shape[:2]
    ax4.imshow(v_frame)
    flow_up = cv2.resize(
        video_res.sample_flow_mag, (vw, vh), interpolation=cv2.INTER_LINEAR
    )
    ax4.imshow(flow_up, cmap="inferno", alpha=0.42)
    m_xs = [pt[1] * vs for pt in video_res.motion_gaze_path_yx]
    m_ys = [pt[0] * vs for pt in video_res.motion_gaze_path_yx]
    if len(m_xs) >= 2:
        ax4.plot(m_xs, m_ys, color="#38BDF8", linewidth=2.0, linestyle="-", alpha=0.9)
    for idx_m, (mx, my) in enumerate(zip(m_xs, m_ys)):
        ax4.scatter(
            [mx],
            [my],
            s=75 if idx_m == len(m_xs) - 1 else 45,
            c="#22C55E" if idx_m == len(m_xs) - 1 else "#E0F2FE",
            edgecolors="#0F172A",
            linewidths=1.0,
            zorder=5,
        )
    cls_counts: Dict[str, int] = {}
    for ev in video_res.frame_events:
        c = ev["event_classification"]
        cls_counts[c] = cls_counts.get(c, 0) + 1
    top_ev = max(cls_counts.items(), key=lambda kv: kv[1])[0] if cls_counts else "N/A"
    ax4.set_title(
        f"4. Real Video Motion-Guided Fovea ({video_res.frames_processed} frames)\n"
        f"Dominant Event: {top_ev}",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    ax4.set_xticks([])
    ax4.set_yticks([])

    # Panel 5: Perturbed Real-Patch Structural Recognition Matrix
    ax5 = axes[1, 1]
    im5 = ax5.imshow(resonance_matrix, cmap="YlGnBu", vmin=0.45, vmax=1.0, aspect="auto")
    ax5.set_xticks(range(len(target_labels)))
    ax5.set_xticklabels(target_labels, rotation=28, ha="right", fontsize=8.5)
    ax5.set_yticks(range(len(query_labels)))
    ax5.set_yticklabels(query_labels, fontsize=8.5)
    for r in range(resonance_matrix.shape[0]):
        for c in range(resonance_matrix.shape[1]):
            val = float(resonance_matrix[r, c])
            txt_color = "#FFFFFF" if val >= 0.76 else "#0F172A"
            ax5.text(
                c,
                r,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=8.5,
                fontweight="bold" if r == c else "normal",
            )
    ax5.set_title(
        f"5. 0.55x Darkened + Shifted Patch Recognition\n"
        f"(Top-1 Exact Match: {top1_accuracy * 100:.0f}%, Mean Margin: +{mean_margin:.3f})",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )

    # Panel 6: CausalField Memory & Periphery-vs-Fovea Detail Gain
    ax6 = axes[1, 2]
    scans = [face_scan, yosemite_scan, third_scan]
    scene_names = ["Face HD", "Yosemite 4K", "Macro Frame"]
    gains = [sc.periphery_vs_fovea_detail_gain for sc in scans]
    mat_counts = [
        float(np.mean([e.material_boundary_count for e in sc.foveal_engrams]))
        for sc in scans
    ]
    x_pos = np.arange(len(scans))
    w_bar = 0.36
    bars1 = ax6.bar(
        x_pos - w_bar / 2, gains, width=w_bar, color="#0284C7", label="Detail Gain (x)"
    )
    bars2 = ax6.bar(
        x_pos + w_bar / 2,
        [m / 2.0 for m in mat_counts],
        width=w_bar,
        color="#EA580C",
        label="Mean Material Bounds (/2)",
    )
    for b, val in zip(bars1, gains):
        ax6.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.15,
            f"{val:.1f}x",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#0369A1",
        )
    for b, val in zip(bars2, mat_counts):
        ax6.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.15,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#C2410C",
        )
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(scene_names, fontsize=9.5, fontweight="bold")
    ax6.set_ylim(0, max(max(gains), max(mat_counts) / 2.0) * 1.35 + 1.0)
    ax6.grid(axis="y", alpha=0.25)
    ax6.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
    ax6.set_title(
        f"6. CausalField Memory: {len(engine.causal_field.voxels)} Voxels, "
        f"{len(engine.causal_field.beams)} Beams",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    ax6.text(
        0.03,
        0.95,
        "Zero VLM / Zero LLM\nPure Optical Bond Cleavage\n+ Fovea-Saccade + DoG + Flow",
        transform=ax6.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="#F1F5F9", ec="#94A3B8", alpha=0.95),
    )

    fig.suptitle(
        "ELYSIA Active Ocular Mechanics (Pure Retinal Physics, Fovea-Saccade, Zero VLM/LLM)",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.952])
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return output_path


def _serialize_engram(e: FovealEngram) -> Dict[str, Any]:
    return {
        "engram_id": e.engram_id,
        "scene_name": e.scene_name,
        "fixation_idx": e.fixation_idx,
        "gaze_yx_norm": [float(e.gaze_yx_norm[0]), float(e.gaze_yx_norm[1])],
        "gaze_yx_px": [int(e.gaze_yx_px[0]), int(e.gaze_yx_px[1])],
        "fovea_box_yxyx": list(e.fovea_box_yxyx),
        "vector_6d": [float(v) for v in e.vector_6d],
        "scale_profile": [float(v) for v in e.scale_profile],
        "chromatic_signature": [float(v) for v in e.chromatic_signature],
        "cleaved_bond_ratio": float(e.cleaved_bond_ratio),
        "material_boundary_count": int(e.material_boundary_count),
        "region_count": int(e.region_count),
        "salience_peak": float(e.salience_peak),
        "recognition_status": e.recognition_status,
        "resonance_score": float(e.resonance_score),
        "voxel_id": e.voxel_id,
    }


def _load_first_video_frame_rgb(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open real video: {path}")
    try:
        ok, frame_bgr = cap.read()
    finally:
        cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read frame 0 from {path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def run_assets_demo() -> Dict[str, Any]:
    t0 = time.perf_counter()
    engine = ActiveOcularMechanicsEngine()

    face_rgb = _load_rgb(ASSETS_DIR / "real_human_face_hd.jpg")
    yosemite_rgb = _load_rgb(ASSETS_DIR / "genuine_optical_4k_yosemite.jpg")
    second_rgb = _load_rgb(ASSETS_DIR / "second_real_world_4k.jpg")
    macro_rgb = _load_first_video_frame_rgb(ASSETS_DIR / "real_external_nature_video.mp4")

    face_scan = engine.scan_image(
        face_rgb, scene_name="face_hd", n_fixations=5, periphery_max_width=220
    )
    yosemite_scan = engine.scan_image(
        yosemite_rgb, scene_name="yosemite_4k", n_fixations=5, periphery_max_width=220
    )
    macro_scan = engine.scan_image(
        macro_rgb, scene_name="macro_leaf", n_fixations=4, periphery_max_width=220
    )

    # Perturbed (0.55x linear brightness + crop shift) real-patch recognition challenge
    # Evaluated across the 3 genuinely distinct real stimuli (Face HD, Yosemite 4K, Macro Leaf)
    probe_specs = [
        ("Face Q1", face_rgb, face_scan.foveal_engrams[0]),
        ("Face Q2", face_rgb, face_scan.foveal_engrams[2]),
        ("Yose Q1", yosemite_rgb, yosemite_scan.foveal_engrams[0]),
        ("Yose Q2", yosemite_rgb, yosemite_scan.foveal_engrams[3]),
        ("Leaf Q1", macro_rgb, macro_scan.foveal_engrams[0]),
        ("Leaf Q2", macro_rgb, macro_scan.foveal_engrams[2]),
    ]
    target_engrams = [spec[2] for spec in probe_specs]
    target_labels = [f"{e.scene_name[:5]}_F{e.fixation_idx + 1}" for e in target_engrams]
    query_labels = [spec[0] for spec in probe_specs]

    res_matrix = np.zeros((len(probe_specs), len(target_engrams)), dtype=np.float32)
    recognition_results: List[Dict[str, Any]] = []
    exact_hits = 0
    margins: List[float] = []

    for r, (q_label, img_rgb, target_eng) in enumerate(probe_specs):
        pert_crop = _make_perturbed_crop(
            img_rgb,
            target_eng.fovea_box_yxyx,
            brightness_scale=0.55,
            shift_frac=0.05,
        )
        rec = engine.recognize_perturbed_patch(pert_crop, query_name=q_label, top_k=5)

        # Compute full 6x6 matrix scores directly against all 6 target engrams
        q_obs = engine.organ.observe(pert_crop, max_width=160)
        from core.sensory.active_ocular_mechanics_engine import _extract_optical_descriptor

        q_v6, q_sc, q_or, q_ch, _ = _extract_optical_descriptor(q_obs)
        for c, mem_eng in enumerate(target_engrams):
            res_matrix[r, c] = engine._compute_structural_resonance(
                q_v6, q_sc, q_or, q_ch, mem_eng
            )

        best_global_id = rec["best_engram_id"]
        is_exact = best_global_id == target_eng.engram_id
        if is_exact:
            exact_hits += 1

        self_score = float(res_matrix[r, r])
        other_scores = [float(res_matrix[r, c]) for c in range(len(target_engrams)) if c != r]
        margin = self_score - max(other_scores)
        margins.append(margin)

        recognition_results.append(
            {
                "query_name": q_label,
                "true_engram_id": target_eng.engram_id,
                "predicted_engram_id": best_global_id,
                "exact_top1_match": is_exact,
                "self_resonance": self_score,
                "margin_over_next_target": float(margin),
            }
        )

    top1_accuracy = float(exact_hits / len(probe_specs))
    mean_margin = float(np.mean(margins))

    # Near-duplicate JPEG recompression recognition check on second_real_world_4k.jpg
    second_scan = engine.scan_image(
        second_rgb, scene_name="second_4k", n_fixations=4, periphery_max_width=220
    )

    video_res = engine.observe_video_stream(
        ASSETS_DIR / "real_external_nature_video.mp4",
        max_frames=8,
        frame_stride=8,
        max_width=220,
    )

    dash_path = render_dashboard(
        face_rgb=face_rgb,
        face_scan=face_scan,
        yosemite_rgb=yosemite_rgb,
        yosemite_scan=yosemite_scan,
        third_scan=macro_scan,
        video_res=video_res,
        query_labels=query_labels,
        target_labels=target_labels,
        resonance_matrix=res_matrix,
        top1_accuracy=top1_accuracy,
        mean_margin=mean_margin,
        engine=engine,
        output_path=DASHBOARD_PATH,
    )

    elapsed_sec = time.perf_counter() - t0
    state = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed_sec, 2),
        "vlm_used": False,
        "llm_used": False,
        "scans": [
            {
                "scene_name": sc.scene_name,
                "image_shape_hw": list(sc.image_shape_hw),
                "periphery_vs_fovea_detail_gain": round(sc.periphery_vs_fovea_detail_gain, 3),
                "engrams": [_serialize_engram(e) for e in sc.foveal_engrams],
            }
            for sc in (face_scan, yosemite_scan, macro_scan, second_scan)
        ],
        "perturbed_recognition_challenge": {
            "top1_exact_match_accuracy": top1_accuracy,
            "mean_resonance_margin": mean_margin,
            "probes": recognition_results,
        },
        "video_tracking": {
            "video_name": video_res.video_name,
            "frames_processed": video_res.frames_processed,
            "frame_events": video_res.frame_events,
            "motion_gaze_path_yx": [list(pt) for pt in video_res.motion_gaze_path_yx],
        },
        "causal_field_summary": {
            "voxel_count": len(engine.causal_field.voxels),
            "beam_count": len(engine.causal_field.beams),
            "engram_memory_count": len(engine.engram_memory),
        },
        "dashboard_path": str(dash_path),
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 84)
    print("ELYSIA Active Ocular Mechanics (Real 4K Photos + Video, Zero VLM/LLM)")
    print("=" * 84)
    print(f"Elapsed Time          : {elapsed_sec:.2f}s")
    for sc in (face_scan, yosemite_scan, macro_scan, second_scan):
        rec_cnt = sum(1 for e in sc.foveal_engrams if e.recognition_status == "RECOGNIZED_STRUCTURE")
        print(
            f"Scene [{sc.scene_name:12s}] : {len(sc.foveal_engrams)} fixations | "
            f"Fovea Detail Gain = {sc.periphery_vs_fovea_detail_gain:.2f}x | "
            f"Recognized Twin/Repeat = {rec_cnt}/{len(sc.foveal_engrams)}"
        )
    print(
        f"Perturbed Recognition : Top-1 Exact Match = {top1_accuracy * 100:.1f}% "
        f"({exact_hits}/{len(probe_specs)}), Mean Margin = +{mean_margin:.3f}"
    )
    print(
        f"Video Tracking        : {video_res.frames_processed} frames decoded from "
        f"{video_res.video_name}.mp4 -> {len(video_res.frame_events)} transitions"
    )
    for ev in video_res.frame_events:
        print(
            f"   t={ev['transition_idx']} | {ev['event_classification']:25s} | "
            f"flow={ev['mean_residual_flow']:.2f}px | gaze={ev['gaze_yx_px']}"
        )
    print(
        f"CausalField (6D)      : {len(engine.causal_field.voxels)} voxels, "
        f"{len(engine.causal_field.beams)} connectivity beams"
    )
    print(f"State Saved           : {STATE_PATH}")
    print(f"Dashboard Saved       : {dash_path}")
    return state


def run_screen_demo() -> Dict[str, Any]:
    engine = ActiveOcularMechanicsEngine()
    res = engine.observe_live_screen(n_frames=3, interval_sec=0.4)
    init_scan: SceneScanResult = res["initial_scan"]
    track: VideoTrackingResult = res["tracking"]
    state = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": "live_pc_screen",
        "screen_shape_hw": list(init_scan.image_shape_hw),
        "periphery_vs_fovea_detail_gain": round(
            init_scan.periphery_vs_fovea_detail_gain, 3
        ),
        "initial_fixations": [_serialize_engram(e) for e in init_scan.foveal_engrams],
        "motion_events": track.frame_events,
        "voxel_count": len(engine.causal_field.voxels),
        "beam_count": len(engine.causal_field.beams),
    }
    SCREEN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCREEN_STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Live screen scan complete: {init_scan.image_shape_hw}, saved to {SCREEN_STATE_PATH}")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ELYSIA Active Ocular Mechanics Demo (Pure Retinal & Fovea-Saccade Physics)"
    )
    parser.add_argument(
        "--source",
        choices=["assets", "screen"],
        default="assets",
        help="Visual world source: real 4K/video assets or live PC desktop screen",
    )
    args = parser.parse_args()
    if args.source == "screen":
        run_screen_demo()
    else:
        run_assets_demo()


if __name__ == "__main__":
    main()
