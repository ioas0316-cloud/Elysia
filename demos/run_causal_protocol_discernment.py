from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.sensory.causal_protocol_discernment_engine import (
    CausalProtocolDiscernmentEngine,
    ProtocolDiscernmentReport,
    ProtocolMeetingComparison,
    stylize_to_cel_anime_digital,
)
from core.sensory.ocular_scale_organ import OcularScaleOrgan

ASSETS_DIR = Path(r"c:\Elysia\demos\real_world_assets")
UI_ASSET_PATH = Path(r"c:\Elysia\docs\assets\elysia_unified_mind_organism_dashboard.png")
STATE_PATH = Path(r"c:\Elysia\data\causal_protocol_discernment_state.json")
DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_causal_protocol_discernment_dashboard.png"
)


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read real image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _short_verdict(v: str) -> str:
    if v == "PHYSICAL_OPTICAL_REALITY":
        return "PHYSICAL"
    if v == "HYBRID_MODULATED_REALITY":
        return "HYBRID"
    return "DIGITAL"


def _serialize_report(r: ProtocolDiscernmentReport) -> Dict[str, Any]:
    o = r.observables
    return {
        "stimulus_name": r.stimulus_name,
        "protocol_verdict": r.protocol_verdict,
        "physical_reality_score": round(r.physical_reality_score, 4),
        "hybrid_modulation_score": round(r.hybrid_modulation_score, 4),
        "digital_realization_score": round(r.digital_realization_score, 4),
        "observables": {
            "flat_plateau_ratio": round(o.flat_plateau_ratio, 4),
            "palette_top12_concentration": round(o.palette_top12_concentration, 4),
            "rectilinear_axis_ratio": round(o.rectilinear_axis_ratio, 4),
            "micro_to_macro_continuity": round(o.micro_to_macro_continuity, 4),
            "edge_bimodality_index": round(o.edge_bimodality_index, 4),
            "chromatic_manifold_deviation": round(o.chromatic_manifold_deviation, 4),
        },
        "explanation_ko": r.explanation_ko,
        "voxel_id": r.voxel_id,
    }


def _serialize_comparison(c: ProtocolMeetingComparison) -> Dict[str, Any]:
    return {
        "name_a": c.name_a,
        "name_b": c.name_b,
        "verdict_a": c.verdict_a,
        "verdict_b": c.verdict_b,
        "macro_meeting_score": round(c.macro_meeting_score, 4),
        "micro_substrate_divergence": round(c.micro_substrate_divergence, 4),
        "scale_resonance_profile": {str(k): round(v, 4) for k, v in c.scale_resonance_profile.items()},
        "meeting_point_summary_ko": c.meeting_point_summary_ko,
    }


def _hstack_equal_height(imgs: List[np.ndarray], target_h: int = 240) -> np.ndarray:
    resized: List[np.ndarray] = []
    for im in imgs:
        h, w = im.shape[:2]
        nw = max(32, int(round(w * (target_h / float(h)))))
        resized.append(cv2.resize(im, (nw, target_h), interpolation=cv2.INTER_AREA))
    gap = np.full((target_h, 6, 3), 240, dtype=np.uint8)
    out_parts: List[np.ndarray] = []
    for i, part in enumerate(resized):
        if i > 0:
            out_parts.append(gap)
        out_parts.append(part)
    return np.concatenate(out_parts, axis=1)


def render_dashboard(
    reports: List[ProtocolDiscernmentReport],
    comparisons: List[ProtocolMeetingComparison],
    cel_face_rgb: np.ndarray,
    mod_sky_rgb: np.ndarray,
    cel_sky_rgb: np.ndarray,
    engine: CausalProtocolDiscernmentEngine,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), dpi=150)
    fig.patch.set_facecolor("#F8FAFC")

    r_map = {r.stimulus_name: r for r in reports}

    # Panel 1: Real Human vs Cel-Anime Human
    ax1 = axes[0, 0]
    p1_img = _hstack_equal_height(
        [r_map["real_face_photo"].obs.rgb_working * 255.0, cel_face_rgb], target_h=240
    )
    ax1.imshow(np.clip(p1_img, 0, 255).astype(np.uint8))
    ax1.set_title(
        f"1. Real Human ({_short_verdict(r_map['real_face_photo'].protocol_verdict)}) vs "
        f"Cel-Anime ({_short_verdict(r_map['anime_cel_face'].protocol_verdict)})\n"
        f"(Macro Meet={comparisons[0].macro_meeting_score:.2f} | "
        f"Micro Substrate Div={comparisons[0].micro_substrate_divergence:.2f})",
        fontsize=9.6,
        fontweight="bold",
        pad=8,
    )
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Panel 2: Real Sky vs Hybrid Neon Sky vs Digital Cel Sky
    ax2 = axes[0, 1]
    p2_img = _hstack_equal_height(
        [r_map["real_yosemite_sky"].obs.rgb_working * 255.0, mod_sky_rgb, cel_sky_rgb],
        target_h=160,
    )
    ax2.imshow(np.clip(p2_img, 0, 255).astype(np.uint8))
    ax2.set_title(
        f"2. Sky Reality Split: {_short_verdict(r_map['real_yosemite_sky'].protocol_verdict)} vs "
        f"{_short_verdict(r_map['modulated_neon_sky'].protocol_verdict)} vs "
        f"{_short_verdict(r_map['anime_cel_sky'].protocol_verdict)}\n"
        f"(Macro Meet Real/Hybrid={comparisons[1].macro_meeting_score:.2f})",
        fontsize=9.6,
        fontweight="bold",
        pad=8,
    )
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Panel 3: Digital UI Graphic
    ax3 = axes[0, 2]
    ui_rep = r_map["digital_ui_dashboard"]
    ax3.imshow(np.clip(ui_rep.obs.rgb_working, 0.0, 1.0))
    ax3.set_title(
        f"3. Digital Graphic Substrate ({_short_verdict(ui_rep.protocol_verdict)})\n"
        f"(Flat={ui_rep.observables.flat_plateau_ratio:.2f}, "
        f"Palette={ui_rep.observables.palette_top12_concentration:.2f}, "
        f"Rect={ui_rep.observables.rectilinear_axis_ratio:.2f})",
        fontsize=9.6,
        fontweight="bold",
        pad=8,
    )
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Panel 4: Multi-Scale Meeting & Divergence Curve (sigma = 1 -> 16)
    ax4 = axes[1, 0]
    scales = list(OcularScaleOrgan.SCALES)
    x_idx = np.arange(len(scales))
    colors = ["#0284C7", "#7C3AED", "#DC2626"]
    markers = ["o", "s", "^"]
    short_pair_names = [
        "Real Face vs Cel Face",
        "Real Sky vs Hybrid Sky",
        "Real Sky vs Digital UI",
    ]
    for comp, col, mk, lbl in zip(comparisons, colors, markers, short_pair_names):
        vals = [comp.scale_resonance_profile[float(s)] for s in scales]
        ax4.plot(x_idx, vals, color=col, marker=mk, lw=2.2, label=lbl)
    ax4.axhline(0.0, color="#94A3B8", lw=0.9, linestyle="--")
    ax4.set_xticks(x_idx)
    ax4.set_xticklabels([f"{int(s)}px" for s in scales], fontsize=9.0, fontweight="bold")
    ax4.set_xlabel("Spatial Observation Scale (Micro 1px -> Macro 16px)", fontsize=8.8)
    ax4.set_ylabel("Structural Energy Correlation", fontsize=8.8)
    ax4.set_ylim(-0.25, 1.15)
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc="lower right", fontsize=8.0, framealpha=0.92)
    ax4.set_title(
        "4. Where Causalities Meet & Split Across Scale\n"
        "(Same Macro Form at 8-16px vs Substrate Split at 1-2px)",
        fontsize=9.8,
        fontweight="bold",
        pad=8,
    )

    # Panel 5: Protocol Observable Fingerprint Heatmap
    ax5 = axes[1, 1]
    obs_cols = ["FlatFill", "PalTop12", "RectAxis", "Bimodality", "ChromaDev"]
    row_labels = [
        "Real Face",
        "Anime Face",
        "Real Sky",
        "Hybrid Sky",
        "Anime Sky",
        "Digital UI",
    ]
    mat = np.array(
        [
            [
                r.observables.flat_plateau_ratio,
                r.observables.palette_top12_concentration,
                r.observables.rectilinear_axis_ratio,
                r.observables.edge_bimodality_index,
                r.observables.chromatic_manifold_deviation,
            ]
            for r in reports
        ],
        dtype=np.float32,
    )
    ax5.imshow(mat, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    ax5.set_xticks(range(len(obs_cols)))
    ax5.set_xticklabels(obs_cols, fontsize=8.5, fontweight="bold")
    ax5.set_yticks(range(len(row_labels)))
    ax5.set_yticklabels(row_labels, fontsize=8.5, fontweight="bold")
    for rr in range(mat.shape[0]):
        for cc in range(mat.shape[1]):
            val = float(mat[rr, cc])
            ax5.text(
                cc,
                rr,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8.2,
                fontweight="bold",
                color="#FFFFFF" if val >= 0.56 else "#0F172A",
            )
    ax5.set_title(
        "5. Blind Physical Protocol Observables\n(Zero Filename or Metadata Hints)",
        fontsize=9.8,
        fontweight="bold",
        pad=8,
    )

    # Panel 6: 3-Regime Continuous Protocol Potential Bars
    ax6 = axes[1, 2]
    x_r = np.arange(len(reports))
    w_b = 0.26
    p_scores = [r.physical_reality_score for r in reports]
    h_scores = [r.hybrid_modulation_score for r in reports]
    d_scores = [r.digital_realization_score for r in reports]

    ax6.bar(x_r - w_b, p_scores, width=w_b, color="#059669", label="Physical Reality")
    ax6.bar(x_r, h_scores, width=w_b, color="#7C3AED", label="Hybrid Modulated")
    ax6.bar(x_r + w_b, d_scores, width=w_b, color="#EA580C", label="Digital Synthetic")
    ax6.set_xticks(x_r)
    ax6.set_xticklabels(row_labels, rotation=22, ha="right", fontsize=8.2, fontweight="bold")
    ax6.set_ylim(0.0, 1.25)
    ax6.grid(axis="y", alpha=0.25)
    ax6.legend(loc="upper left", fontsize=7.8, ncol=2, framealpha=0.92)
    ax6.set_title(
        f"6. Protocol Regime Potentials ({len(engine.causal_field.voxels)} Voxels, "
        f"{len(engine.causal_field.beams)} Beams)",
        fontsize=9.8,
        fontweight="bold",
        pad=8,
    )

    fig.suptitle(
        "ELYSIA Causal Ontological Protocol Discernment: Realized Reality vs Modulated vs Digital",
        fontsize=13.4,
        fontweight="bold",
        y=0.982,
    )
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.945], h_pad=2.4)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return output_path


def run_demo() -> Dict[str, Any]:
    t0 = time.perf_counter()
    engine = CausalProtocolDiscernmentEngine()

    face_rgb = _load_rgb(ASSETS_DIR / "real_human_face_hd.jpg")
    yose_rgb = _load_rgb(ASSETS_DIR / "genuine_optical_4k_yosemite.jpg")
    ui_rgb = _load_rgb(UI_ASSET_PATH)

    cel_face_rgb = stylize_to_cel_anime_digital(face_rgb, n_colors=10, max_width=480)
    cel_sky_rgb = stylize_to_cel_anime_digital(yose_rgb, n_colors=10, max_width=480)

    yose_fact = engine.gen.factorize(yose_rgb, source_name="yose_mod_src", max_width=320)
    mod_sky_res = engine.gen.modulate(
        yose_fact,
        name="modulated_neon_sky",
        chroma_rotation_deg=110.0,
        chroma_gain=1.18,
    )
    mod_sky_rgb = mod_sky_res.synthesized_rgb

    rep_real_face = engine.discern_protocol(face_rgb, "real_face_photo", max_width=240)
    rep_cel_face = engine.discern_protocol(cel_face_rgb, "anime_cel_face", max_width=240)
    rep_real_sky = engine.discern_protocol(yose_rgb, "real_yosemite_sky", max_width=240)
    rep_mod_sky = engine.discern_protocol(mod_sky_rgb, "modulated_neon_sky", max_width=240)
    rep_cel_sky = engine.discern_protocol(cel_sky_rgb, "anime_cel_sky", max_width=240)
    rep_ui = engine.discern_protocol(ui_rgb, "digital_ui_dashboard", max_width=240)

    reports = [
        rep_real_face,
        rep_cel_face,
        rep_real_sky,
        rep_mod_sky,
        rep_cel_sky,
        rep_ui,
    ]

    cmp_face = engine.compare_protocol_meeting(rep_real_face, rep_cel_face)
    cmp_sky_mod = engine.compare_protocol_meeting(rep_real_sky, rep_mod_sky)
    cmp_sky_ui = engine.compare_protocol_meeting(rep_real_sky, rep_ui)
    comparisons = [cmp_face, cmp_sky_mod, cmp_sky_ui]

    dash_path = render_dashboard(
        reports=reports,
        comparisons=comparisons,
        cel_face_rgb=cel_face_rgb,
        mod_sky_rgb=mod_sky_rgb,
        cel_sky_rgb=cel_sky_rgb,
        engine=engine,
        output_path=DASHBOARD_PATH,
    )

    elapsed_sec = time.perf_counter() - t0
    state = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed_sec, 2),
        "vlm_used": False,
        "llm_used": False,
        "reports": [_serialize_report(r) for r in reports],
        "comparisons": [_serialize_comparison(c) for c in comparisons],
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
    print("ELYSIA Causal Ontological Protocol Discernment (Zero VLM/LLM, Zero Hints)")
    print("=" * 92)
    print(f"Elapsed Time          : {elapsed_sec:.2f}s")
    for r in reports:
        o = r.observables
        print(
            f"  [{r.stimulus_name:20s}] -> {r.protocol_verdict:27s} | "
            f"Phys={r.physical_reality_score:.2f} Hyb={r.hybrid_modulation_score:.2f} Dig={r.digital_realization_score:.2f} | "
            f"(Flat={o.flat_plateau_ratio:.2f} Pal={o.palette_top12_concentration:.2f} "
            f"Rect={o.rectilinear_axis_ratio:.2f} Bim={o.edge_bimodality_index:.2f} Chr={o.chromatic_manifold_deviation:.2f})"
        )
    print("\n[Protocol Meeting & Divergence Across Scales]")
    for c in comparisons:
        print(
            f"  - {c.name_a} vs {c.name_b}: macro_meet={c.macro_meeting_score:.3f}, "
            f"micro_div={c.micro_substrate_divergence:.3f}"
        )
        print(f"    -> {c.meeting_point_summary_ko}")
    print(
        f"\nCausalField (6D)      : {len(engine.causal_field.voxels)} voxels, "
        f"{len(engine.causal_field.beams)} connectivity beams"
    )
    print(f"State Saved           : {STATE_PATH}")
    print(f"Dashboard Saved       : {dash_path}")
    return state


if __name__ == "__main__":
    run_demo()
