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

from core.sensory.causal_protocol_discernment_engine import stylize_to_cel_anime_digital
from core.sensory.functional_relational_hierarchy_engine import (
    BehavioralAgencyTrace,
    FunctionalRelationalHierarchyEngine,
    StratifiedEntityJudgment,
    render_minimalist_cartoon_animal,
    render_stone_effigy_from_photo,
)

ASSETS_DIR = Path(r"c:\Elysia\demos\real_world_assets")
STATE_PATH = Path(r"c:\Elysia\data\functional_relational_hierarchy_state.json")
DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_functional_relational_hierarchy_dashboard.png"
)


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read real image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _short_entity(v: str) -> str:
    return {
        "LIVING_ANIMAL_AGENT": "LIVING AGENT",
        "CHARACTER_REPRESENTATION": "CHARACTER",
        "INANIMATE_EFFIGY": "STATUE/EFFIGY",
        "NON_AGENT_ENVIRONMENT": "ENVIRONMENT",
    }.get(v, v)


def _short_proto(v: str) -> str:
    if v == "PHYSICAL_OPTICAL_REALITY":
        return "PHYSICAL"
    if v == "HYBRID_MODULATED_REALITY":
        return "HYBRID"
    return "DIGITAL"


def _draw_judgment_panel(ax: plt.Axes, j: StratifiedEntityJudgment, title_prefix: str) -> None:
    h, w = j.obs.shape_hw
    ax.imshow(np.clip(j.obs.rgb_working, 0.0, 1.0))

    for org in j.detected_organs:
        y1 = org.anchor_yx_norm[0] * (h - 1)
        x1 = org.anchor_yx_norm[1] * (w - 1)
        if org.organ_type == "EYE_PAIR":
            ax.scatter([x1], [y1], s=75, c="#22D3EE", edgecolors="#0F172A", lw=1.2, zorder=5)
            if org.secondary_yx_norm is not None:
                y2 = org.secondary_yx_norm[0] * (h - 1)
                x2 = org.secondary_yx_norm[1] * (w - 1)
                ax.scatter([x2], [y2], s=75, c="#22D3EE", edgecolors="#0F172A", lw=1.2, zorder=5)
                ax.plot([x1, x2], [y1, y2], color="#22D3EE", lw=1.5, ls="--", alpha=0.85, zorder=4)
        elif org.organ_type == "NOSE":
            ax.scatter([x1], [y1], s=65, marker="D", c="#FACC15", edgecolors="#0F172A", lw=1.0, zorder=5)
        elif org.organ_type == "MOUTH":
            ax.scatter([x1], [y1], s=70, marker="s", c="#FB7185", edgecolors="#0F172A", lw=1.0, zorder=5)
        elif org.organ_type == "EAR_PAIR":
            ax.scatter([x1], [y1], s=60, marker="^", c="#4ADE80", edgecolors="#0F172A", lw=1.0, zorder=5)
            if org.secondary_yx_norm is not None:
                y2 = org.secondary_yx_norm[0] * (h - 1)
                x2 = org.secondary_yx_norm[1] * (w - 1)
                ax.scatter([x2], [y2], s=60, marker="^", c="#4ADE80", edgecolors="#0F172A", lw=1.0, zorder=5)

    iso_str = (
        f"FuncIso={j.functional_isomorphism_to_reference:.2f}"
        if j.functional_isomorphism_to_reference is not None
        else "Ref Anchor"
    )
    pix_str = (
        f"Pix={j.pixel_similarity_to_reference:.2f}"
        if j.pixel_similarity_to_reference is not None
        else "Pix=1.00"
    )
    ax.set_title(
        f"{title_prefix}: {_short_entity(j.entity_verdict)} [{_short_proto(j.protocol_report.protocol_verdict)}]\n"
        f"(Triad={j.organ_triad_coherence:.2f} | Ag={j.agency_score:.2f} | {iso_str} | {pix_str})",
        fontsize=9.4,
        fontweight="bold",
        pad=7,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _serialize_judgment(j: StratifiedEntityJudgment) -> Dict[str, Any]:
    return {
        "stimulus_name": j.stimulus_name,
        "entity_verdict": j.entity_verdict,
        "protocol_verdict": j.protocol_report.protocol_verdict,
        "organ_triad_coherence": round(j.organ_triad_coherence, 4),
        "agency_score": round(j.agency_score, 4),
        "pixel_similarity_to_reference": (
            round(j.pixel_similarity_to_reference, 4)
            if j.pixel_similarity_to_reference is not None
            else None
        ),
        "functional_isomorphism_to_reference": (
            round(j.functional_isomorphism_to_reference, 4)
            if j.functional_isomorphism_to_reference is not None
            else None
        ),
        "detected_organs": [
            {
                "organ_type": o.organ_type,
                "language_label_ko": o.language_label_ko,
                "confidence": round(o.relational_confidence, 4),
                "anchor_yx_norm": [round(o.anchor_yx_norm[0], 4), round(o.anchor_yx_norm[1], 4)],
            }
            for o in j.detected_organs
        ],
        "explanation_ko": j.explanation_ko,
        "entity_voxel_id": j.entity_voxel_id,
    }


def render_dashboard(
    judgments: List[StratifiedEntityJudgment],
    engine: FunctionalRelationalHierarchyEngine,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), dpi=150)
    fig.patch.set_facecolor("#F8FAFC")

    _draw_judgment_panel(axes[0, 0], judgments[0], "1. Real Human")
    _draw_judgment_panel(axes[0, 1], judgments[1], "2. Cel-Anime Human")
    _draw_judgment_panel(axes[0, 2], judgments[2], "3. Minimalist 2-Dot Animal")
    _draw_judgment_panel(axes[1, 0], judgments[3], "4. Lifeless Stone Statue")
    _draw_judgment_panel(axes[1, 1], judgments[4], "5. Yosemite Landscape")

    # Panel 6: Flat Pixel Similarity vs 4-Tier Functional Isomorphism & Agency
    ax6 = axes[1, 2]
    labels = ["Real\nHuman", "Cel\nHuman", "2-Dot\nAnimal", "Stone\nStatue", "Yosemite\nLand"]
    pix_vals = [
        1.0 if j.pixel_similarity_to_reference is None else j.pixel_similarity_to_reference
        for j in judgments
    ]
    func_vals = [
        1.0 if j.functional_isomorphism_to_reference is None else j.functional_isomorphism_to_reference
        for j in judgments
    ]
    ag_vals = [j.agency_score for j in judgments]

    x = np.arange(len(judgments))
    w = 0.25
    b1 = ax6.bar(x - w, pix_vals, width=w, color="#94A3B8", label="Flat Pixel Sim")
    b2 = ax6.bar(x, func_vals, width=w, color="#0284C7", label="Functional Role Iso")
    b3 = ax6.bar(x + w, ag_vals, width=w, color="#16A34A", label="Causal Agency")

    for bars, col in ((b1, "#475569"), (b2, "#0369A1"), (b3, "#15803D")):
        for bar in bars:
            val = float(bar.get_height())
            ax6.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.015,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color=col,
            )

    ax6.set_xticks(x)
    ax6.set_xticklabels(labels, fontsize=8.5, fontweight="bold")
    ax6.set_ylim(0.0, 1.28)
    ax6.grid(axis="y", alpha=0.25)
    ax6.legend(loc="upper right", fontsize=7.6, framealpha=0.92)
    ax6.set_title(
        f"6. Why Pixels Alone Fail vs 4-Tier Hierarchy\n"
        f"CausalField: {len(engine.causal_field.voxels)} Voxels, "
        f"{len(engine.causal_field.beams)} Beams",
        fontsize=9.6,
        fontweight="bold",
        pad=8,
    )

    fig.suptitle(
        "ELYSIA 4-Tier Stratified Discernment: Sensory -> Functional Organs (Lang) -> Entity/Agency -> Protocol",
        fontsize=13.0,
        fontweight="bold",
        y=0.982,
    )
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.945], h_pad=2.3)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return output_path


def run_demo() -> Dict[str, Any]:
    t0 = time.perf_counter()
    engine = FunctionalRelationalHierarchyEngine()

    face_rgb = _load_rgb(ASSETS_DIR / "real_human_face_hd.jpg")
    yose_rgb = _load_rgb(ASSETS_DIR / "genuine_optical_4k_yosemite.jpg")
    cel_face_rgb = stylize_to_cel_anime_digital(face_rgb, n_colors=10, max_width=480)
    dot_animal_rgb = render_minimalist_cartoon_animal((280, 280))
    stone_statue_rgb = render_stone_effigy_from_photo(face_rgb)

    j_real = engine.judge_entity(
        face_rgb,
        "real_human_face",
        BehavioralAgencyTrace(0.92, 0.88, 0.85, 0.95),
        max_width=240,
    )
    j_cel = engine.judge_entity(
        cel_face_rgb,
        "cel_anime_human",
        BehavioralAgencyTrace(0.75, 0.70, 0.80, 0.75),
        reference_name="real_human_face",
        max_width=240,
    )
    j_dot = engine.judge_entity(
        dot_animal_rgb,
        "minimalist_dot_animal",
        BehavioralAgencyTrace(0.75, 0.75, 0.80, 0.70),
        reference_name="real_human_face",
        max_width=240,
    )
    j_stone = engine.judge_entity(
        stone_statue_rgb,
        "lifeless_stone_statue",
        BehavioralAgencyTrace(0.0, 0.0, 0.0, 0.0),
        reference_name="real_human_face",
        max_width=240,
    )
    j_yose = engine.judge_entity(
        yose_rgb,
        "yosemite_landscape",
        BehavioralAgencyTrace(0.0, 0.0, 0.0, 0.0),
        reference_name="real_human_face",
        max_width=240,
    )

    judgments = [j_real, j_cel, j_dot, j_stone, j_yose]
    dash_path = render_dashboard(judgments, engine, DASHBOARD_PATH)

    elapsed_sec = time.perf_counter() - t0
    state = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed_sec, 2),
        "vlm_used": False,
        "llm_used": False,
        "judgments": [_serialize_judgment(j) for j in judgments],
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
    print("ELYSIA 4-Tier Functional-Relational Stratified Discernment (Zero VLM/LLM)")
    print("=" * 92)
    print(f"Elapsed Time          : {elapsed_sec:.2f}s")
    for j in judgments:
        orgs = ", ".join(o.organ_type for o in j.detected_organs) or "NONE"
        pix_s = f"{j.pixel_similarity_to_reference:.3f}" if j.pixel_similarity_to_reference is not None else "1.000"
        iso_s = f"{j.functional_isomorphism_to_reference:.3f}" if j.functional_isomorphism_to_reference is not None else "1.000"
        print(
            f"  [{j.stimulus_name:22s}] Entity={j.entity_verdict:24s} | "
            f"Proto={_short_proto(j.protocol_report.protocol_verdict):8s} | "
            f"Triad={j.organ_triad_coherence:.2f} | Ag={j.agency_score:.2f} | "
            f"Pix={pix_s} vs FuncIso={iso_s} | Organs=[{orgs}]"
        )
        print(f"     -> {j.explanation_ko}")
    print(
        f"CausalField (6D)      : {len(engine.causal_field.voxels)} voxels, "
        f"{len(engine.causal_field.beams)} connectivity beams"
    )
    print(f"State Saved           : {STATE_PATH}")
    print(f"Dashboard Saved       : {dash_path}")
    return state


if __name__ == "__main__":
    run_demo()
