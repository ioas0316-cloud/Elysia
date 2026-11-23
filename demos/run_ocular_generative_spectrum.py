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

from core.sensory.ocular_generative_spectrum_engine import (
    GenerativeVerificationResult,
    OcularCognitiveFactorization,
    OcularGenerativeSpectrumEngine,
)

ASSETS_DIR = Path(r"c:\Elysia\demos\real_world_assets")
STATE_PATH = Path(r"c:\Elysia\data\ocular_generative_spectrum_state.json")
DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_ocular_generative_spectrum_dashboard.png"
)


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read real image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


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


def _serialize_op(
    res: GenerativeVerificationResult,
    src_fact: OcularCognitiveFactorization,
    donor_fact: OcularCognitiveFactorization | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "operation_name": res.operation_name,
        "mode": res.mode,
        "source_name": src_fact.source_name,
        "structural_corr_with_source": round(res.structural_corr_with_source, 4),
        "optical_resonance_with_source": round(res.optical_resonance_with_source, 4),
        "delta_luma_mean": round(res.delta_luma_mean, 4),
        "delta_chroma_mean": round(res.delta_chroma_mean, 4),
        "psnr_db": round(res.psnr_db, 2) if res.psnr_db is not None else None,
        "fine_energy_before": round(
            float(src_fact.scale_band_energies[0] + src_fact.scale_band_energies[1]), 4
        ),
        "fine_energy_after": round(
            float(res.reobserved_scale_profile[0] + res.reobserved_scale_profile[1]), 4
        ),
        "voxel_id": res.voxel_id,
    }
    if donor_fact is not None:
        before_dist = float(
            np.linalg.norm(
                src_fact.chromatic_signature - donor_fact.chromatic_signature
            )
        )
        after_dist = float(
            np.linalg.norm(
                res.reobserved_chromatic_signature - donor_fact.chromatic_signature
            )
        )
        out["palette_donor_name"] = donor_fact.source_name
        out["optical_resonance_with_donor"] = (
            round(res.optical_resonance_with_palette_donor, 4)
            if res.optical_resonance_with_palette_donor is not None
            else None
        )
        out["chroma_dist_before_to_donor"] = round(before_dist, 4)
        out["chroma_dist_after_to_donor"] = round(after_dist, 4)
    return out


def _show_img_panel(
    ax: plt.Axes,
    rgb_img: np.ndarray,
    title: str,
) -> None:
    ax.imshow(rgb_img)
    ax.set_title(title, fontsize=10.0, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])


def render_dashboard(
    face_fact: OcularCognitiveFactorization,
    face_recon: GenerativeVerificationResult,
    face_painterly: GenerativeVerificationResult,
    rec_face_yose: GenerativeVerificationResult,
    yose_fact: OcularCognitiveFactorization,
    yose_autumn: GenerativeVerificationResult,
    rec_yose_leaf: GenerativeVerificationResult,
    all_ops: List[GenerativeVerificationResult],
    engine: OcularGenerativeSpectrumEngine,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5), dpi=150)
    fig.patch.set_facecolor("#F8FAFC")

    # Row 1: Face Input -> Faithful Reconstruction -> Painterly Spectrum Dial -> Cross-Scene Recreation
    _show_img_panel(
        axes[0, 0],
        face_fact.source_rgb_working_u8,
        f"1. Retinal Input: Face HD\n(Regions={len(face_fact.region_stats)}, FineE="
        f"{face_fact.scale_band_energies[0] + face_fact.scale_band_energies[1]:.2f})",
    )
    _show_img_panel(
        axes[0, 1],
        face_recon.synthesized_rgb,
        f"2. Inverse Axis-Spectrum Repro\n(PSNR={face_recon.psnr_db:.1f}dB, "
        f"StructCorr={face_recon.structural_corr_with_source:.3f}, "
        f"Res={face_recon.optical_resonance_with_source:.3f})",
    )
    _show_img_panel(
        axes[0, 2],
        face_painterly.synthesized_rgb,
        f"3. Spectrum Dial: Painterly\n(B0/B1 suppressed, FineE="
        f"{face_painterly.reobserved_scale_profile[0] + face_painterly.reobserved_scale_profile[1]:.2f}, "
        f"Corr={face_painterly.structural_corr_with_source:.2f})",
    )
    _show_img_panel(
        axes[0, 3],
        rec_face_yose.synthesized_rgb,
        f"4. Recreate: Face Mold x Yosemite\n(StructCorr={rec_face_yose.structural_corr_with_source:.2f}, "
        f"DonorRes={rec_face_yose.optical_resonance_with_palette_donor:.2f})",
    )

    # Row 2: Yosemite Input -> Chromatic Opponent Rotation -> Yosemite x Leaf Recreation -> Closed-Loop Chart
    _show_img_panel(
        axes[1, 0],
        yose_fact.source_rgb_working_u8,
        f"5. Retinal Input: Yosemite 4K\n(Regions={len(yose_fact.region_stats)}, FineE="
        f"{yose_fact.scale_band_energies[0] + yose_fact.scale_band_energies[1]:.2f})",
    )
    _show_img_panel(
        axes[1, 1],
        yose_autumn.synthesized_rgb,
        f"6. Chromatic Axis Rot (+110 deg)\n(dChroma={yose_autumn.delta_chroma_mean:.2f}, "
        f"StructCorr={yose_autumn.structural_corr_with_source:.3f})",
    )
    _show_img_panel(
        axes[1, 2],
        rec_yose_leaf.synthesized_rgb,
        f"7. Recreate: Yosemite Mold x Leaf\n(StructCorr={rec_yose_leaf.structural_corr_with_source:.2f}, "
        f"dChroma={rec_yose_leaf.delta_chroma_mean:.2f})",
    )

    # Panel 8: Closed-Loop Ocular Self-Verification Chart
    ax8 = axes[1, 3]
    short_names = [
        "Face\nRecon",
        "Yose\nRecon",
        "Yose\nDark",
        "Yose\nRot110",
        "Face\nPaint",
        "Face\nRelief",
        "Face\nxYose",
        "Yose\nxLeaf",
    ]
    struct_vals = [op.structural_corr_with_source for op in all_ops]
    chroma_vals = [min(1.0, op.delta_chroma_mean) for op in all_ops]
    luma_vals = [min(1.0, abs(op.delta_luma_mean)) for op in all_ops]

    x = np.arange(len(all_ops))
    w = 0.27
    ax8.bar(x - w, struct_vals, width=w, color="#0284C7", label="Struct Corr (Form)")
    ax8.bar(x, chroma_vals, width=w, color="#EA580C", label="Delta Chroma (Color)")
    ax8.bar(x + w, luma_vals, width=w, color="#7C3AED", label="|Delta Luma| (Light)")
    ax8.axhline(0.75, color="#16A34A", linestyle="--", linewidth=1.0, alpha=0.75)

    ax8.set_xticks(x)
    ax8.set_xticklabels(short_names, fontsize=7.8, fontweight="bold")
    ax8.set_ylim(0.0, 1.32)
    ax8.grid(axis="y", alpha=0.25)
    ax8.legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.92)
    ax8.set_title(
        f"8. Closed-Loop Re-Observation Verify\n"
        f"CausalField: {len(engine.causal_field.voxels)} Voxels, "
        f"{len(engine.causal_field.beams)} Beams (Zero VLM/LLM)",
        fontsize=9.8,
        fontweight="bold",
        pad=6,
    )

    fig.suptitle(
        "ELYSIA Ocular Axis-Spectrum Factorization, Modulation & Cross-Scene Recreation (Zero VLM/LLM)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return output_path


def run_demo() -> Dict[str, Any]:
    t0 = time.perf_counter()
    engine = OcularGenerativeSpectrumEngine()

    face_rgb = _load_rgb(ASSETS_DIR / "real_human_face_hd.jpg")
    yose_rgb = _load_rgb(ASSETS_DIR / "genuine_optical_4k_yosemite.jpg")
    leaf_rgb = _load_first_video_frame_rgb(ASSETS_DIR / "real_external_nature_video.mp4")

    # Stage 1: Axis-Spectrum Factorization
    face_fact = engine.factorize(face_rgb, source_name="face_hd", max_width=240)
    yose_fact = engine.factorize(yose_rgb, source_name="yosemite_4k", max_width=240)
    leaf_fact = engine.factorize(leaf_rgb, source_name="macro_leaf", max_width=240)

    # Stage 2: Faithful Inverse Retinal Reproduction
    face_recon = engine.reconstruct(face_fact, name="face_recon")
    yose_recon = engine.reconstruct(yose_fact, name="yose_recon")

    # Stage 3: Orthogonal Axis & Spectral Modulation
    yose_dark = engine.modulate(
        yose_fact,
        name="yose_twilight_illumination",
        luma_shift=-0.55,
    )
    yose_autumn = engine.modulate(
        yose_fact,
        name="yose_autumn_chroma_rot110",
        chroma_rotation_deg=110.0,
        chroma_gain=1.15,
    )
    face_painterly = engine.modulate(
        face_fact,
        name="face_painterly_macro_spectrum",
        band_gains=(0.05, 0.15, 0.70, 1.15, 1.15),
    )
    face_relief = engine.modulate(
        face_fact,
        name="face_relief_micro_spectrum",
        band_gains=(2.2, 1.8, 1.1, 1.0, 1.0),
    )

    # Stage 4: Cross-Scene Structural Mold x Chromatic/Scale Spectrum Recreation
    rec_face_yose = engine.recreate_cross_scene(
        structure_fact=face_fact,
        palette_fact=yose_fact,
        name="face_mold_x_yosemite_spectrum",
        chroma_blend=0.88,
        spectrum_blend=0.55,
        luma_mood_blend=0.30,
    )
    rec_yose_leaf = engine.recreate_cross_scene(
        structure_fact=yose_fact,
        palette_fact=leaf_fact,
        name="yosemite_mold_x_leaf_spectrum",
        chroma_blend=0.88,
        spectrum_blend=0.55,
        luma_mood_blend=0.30,
    )

    all_ops = [
        face_recon,
        yose_recon,
        yose_dark,
        yose_autumn,
        face_painterly,
        face_relief,
        rec_face_yose,
        rec_yose_leaf,
    ]

    dash_path = render_dashboard(
        face_fact=face_fact,
        face_recon=face_recon,
        face_painterly=face_painterly,
        rec_face_yose=rec_face_yose,
        yose_fact=yose_fact,
        yose_autumn=yose_autumn,
        rec_yose_leaf=rec_yose_leaf,
        all_ops=all_ops,
        engine=engine,
        output_path=DASHBOARD_PATH,
    )

    elapsed_sec = time.perf_counter() - t0
    state = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed_sec, 2),
        "vlm_used": False,
        "llm_used": False,
        "neural_diffusion_used": False,
        "sources": [
            {
                "source_name": f.source_name,
                "shape_hw": list(f.shape_hw),
                "region_count": len(f.region_stats),
                "scale_band_energies": [round(float(v), 4) for v in f.scale_band_energies],
                "chromatic_signature": [round(float(v), 4) for v in f.chromatic_signature],
                "voxel_id": f.voxel_id,
            }
            for f in (face_fact, yose_fact, leaf_fact)
        ],
        "operations": [
            _serialize_op(face_recon, face_fact),
            _serialize_op(yose_recon, yose_fact),
            _serialize_op(yose_dark, yose_fact),
            _serialize_op(yose_autumn, yose_fact),
            _serialize_op(face_painterly, face_fact),
            _serialize_op(face_relief, face_fact),
            _serialize_op(rec_face_yose, face_fact, yose_fact),
            _serialize_op(rec_yose_leaf, yose_fact, leaf_fact),
        ],
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

    print("\n" + "=" * 88)
    print("ELYSIA Ocular Axis-Spectrum Reproduction, Modulation & Recreation (Zero VLM/LLM)")
    print("=" * 88)
    print(f"Elapsed Time          : {elapsed_sec:.2f}s")
    for op_info in state["operations"]:
        psnr_str = f"PSNR={op_info['psnr_db']:.2f}dB | " if op_info["psnr_db"] is not None else ""
        donor_str = (
            f" | donor_dist {op_info['chroma_dist_before_to_donor']:.3f} -> {op_info['chroma_dist_after_to_donor']:.3f}"
            if "chroma_dist_after_to_donor" in op_info
            else ""
        )
        print(
            f"  [{op_info['mode']:23s}] {op_info['operation_name']:30s} | "
            f"{psnr_str}struct={op_info['structural_corr_with_source']:.3f} | "
            f"res={op_info['optical_resonance_with_source']:.3f} | "
            f"dA={op_info['delta_luma_mean']:+.3f} | dC={op_info['delta_chroma_mean']:.3f}"
            f"{donor_str}"
        )
    print(
        f"CausalField (6D)      : {len(engine.causal_field.voxels)} voxels, "
        f"{len(engine.causal_field.beams)} connectivity beams"
    )
    print(f"State Saved           : {STATE_PATH}")
    print(f"Dashboard Saved       : {dash_path}")
    return state


if __name__ == "__main__":
    run_demo()
