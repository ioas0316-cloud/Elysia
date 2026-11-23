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

from core.sensory.ocular_topological_composition_engine import (
    AtomPlacement,
    ComposedSceneResult,
    OcularPrimitiveAtom,
    OcularTopologicalCompositionEngine,
)

ASSETS_DIR = Path(r"c:\Elysia\demos\real_world_assets")
STATE_PATH = Path(r"c:\Elysia\data\ocular_topological_composition_state.json")
DASHBOARD_PATH = Path(
    r"c:\Elysia\docs\assets\elysia_ocular_topological_composition_dashboard.png"
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


def _build_atoms_mosaic(atoms: List[OcularPrimitiveAtom], tile_px: int = 96) -> np.ndarray:
    """Render a 2x3 mosaic of harvested real atoms modulated by their organic support masks."""
    bg = np.array([28.0, 34.0, 44.0], dtype=np.float32)
    tiles: List[np.ndarray] = []
    for atom in atoms[:6]:
        src = cv2.resize(
            atom.factorization.source_rgb_working_u8,
            (tile_px, tile_px),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        m = cv2.resize(
            atom.organic_mask, (tile_px, tile_px), interpolation=cv2.INTER_LINEAR
        )[..., None]
        blended = (1.0 - m) * bg + m * src
        tiles.append(np.clip(blended, 0.0, 255.0).astype(np.uint8))

    gap = 4
    row1 = np.concatenate(
        [
            tiles[0],
            np.full((tile_px, gap, 3), 230, dtype=np.uint8),
            tiles[1],
            np.full((tile_px, gap, 3), 230, dtype=np.uint8),
            tiles[2],
        ],
        axis=1,
    )
    row2 = np.concatenate(
        [
            tiles[3],
            np.full((tile_px, gap, 3), 230, dtype=np.uint8),
            tiles[4],
            np.full((tile_px, gap, 3), 230, dtype=np.uint8),
            tiles[5],
        ],
        axis=1,
    )
    h_sep = np.full((gap, row1.shape[1], 3), 230, dtype=np.uint8)
    return np.concatenate([row1, h_sep, row2], axis=0)


def _render_cleaved_bonds(comp: ComposedSceneResult) -> np.ndarray:
    """Visualize closed-loop cleaved optical bonds on the newly composed scene."""
    obs = comp.reobserved_obs
    h, w = obs.shape_hw
    base = cv2.resize(
        comp.synthesized_rgb, (w, h), interpolation=cv2.INTER_AREA
    ).astype(np.float32) * 0.45

    hc = obs.l2.bond_lattice.horiz_cleaved
    vc = obs.l2.bond_lattice.vert_cleaved
    edges = np.zeros((h, w), dtype=bool)
    edges[:, :-1] |= hc
    edges[:, 1:] |= hc
    edges[:-1, :] |= vc
    edges[1:, :] |= vc

    mat = obs.cause_map == 3
    base[edges] = np.array([34.0, 211.0, 238.0], dtype=np.float32)
    base[edges & mat] = np.array([249.0, 115.0, 22.0], dtype=np.float32)
    return np.clip(base, 0.0, 255.0).astype(np.uint8)


def _serialize_comp(res: ComposedSceneResult) -> Dict[str, Any]:
    mean_recall = float(
        np.mean([r["self_resonance"] for r in res.atom_provenance_recalls])
    )
    mean_distractor = float(
        np.mean([r["mean_non_placed_distractor"] for r in res.atom_provenance_recalls])
    )
    max_src_corr = max(abs(v) for v in res.novelty_vs_sources.values())
    return {
        "composition_name": res.composition_name,
        "canvas_shape_hw": list(res.canvas_shape_hw),
        "atom_count": len(res.placements),
        "cleaved_bond_ratio": round(res.cleaved_bond_ratio, 4),
        "material_boundary_count": res.material_boundary_count,
        "region_count": res.region_count,
        "novelty_vs_sources": {k: round(v, 4) for k, v in res.novelty_vs_sources.items()},
        "max_source_persistence_corr": round(max_src_corr, 4),
        "mean_atom_provenance_resonance": round(mean_recall, 4),
        "mean_non_placed_distractor_resonance": round(mean_distractor, 4),
        "atom_provenance_recalls": res.atom_provenance_recalls,
        "voxel_id": res.voxel_id,
    }


def render_dashboard(
    sample_atoms: List[OcularPrimitiveAtom],
    comp_a: ComposedSceneResult,
    comp_b: ComposedSceneResult,
    comp_c: ComposedSceneResult,
    engine: OcularTopologicalCompositionEngine,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), dpi=150)
    fig.patch.set_facecolor("#F8FAFC")

    # Panel 1: Harvested Optical Primitive Atoms Mosaic
    ax1 = axes[0, 0]
    ax1.imshow(_build_atoms_mosaic(sample_atoms, tile_px=96))
    ax1.set_title(
        f"1. Harvested Organic-Masked Atoms\n"
        f"({len(engine.atoms)} total across Face / Yosemite / Macro Leaf)",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Panel 2: Novel Composition A
    ax2 = axes[0, 1]
    ax2.imshow(comp_a.synthesized_rgb)
    mean_rec_a = float(np.mean([r["self_resonance"] for r in comp_a.atom_provenance_recalls]))
    max_corr_a = max(abs(v) for v in comp_a.novelty_vs_sources.values())
    ax2.set_title(
        f"2. Novel Scene A: Botanical Sanctuary\n"
        f"(Blank Canvas | Recall={mean_rec_a:.2f} | MaxSrcCorr={max_corr_a:.2f})",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Panel 3: Novel Composition B
    ax3 = axes[0, 2]
    ax3.imshow(comp_b.synthesized_rgb)
    mean_rec_b = float(np.mean([r["self_resonance"] for r in comp_b.atom_provenance_recalls]))
    max_corr_b = max(abs(v) for v in comp_b.novelty_vs_sources.values())
    ax3.set_title(
        f"3. Novel Scene B: Portrait in Alpine Mist\n"
        f"(Cross-Domain Harmonized | Recall={mean_rec_b:.2f} | SrcCorr={max_corr_b:.2f})",
        fontsize=10.5,
        fontweight="bold",
        pad=8,
    )
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Panel 4: Parametric Variant C
    ax4 = axes[1, 0]
    ax4.imshow(comp_c.synthesized_rgb)
    mean_rec_c = float(np.mean([r["self_resonance"] for r in comp_c.atom_provenance_recalls]))
    ax4.set_title(
        f"4. Parametric Variant C of Scene A\n"
        f"(Layout Shift + Luma -0.40 + Chroma Rot +45 deg | Recall={mean_rec_c:.2f})",
        fontsize=10.0,
        fontweight="bold",
        pad=8,
    )
    ax4.set_xticks([])
    ax4.set_yticks([])

    # Panel 5: Closed-Loop Re-Observation Cleaved Bonds on Scene B
    ax5 = axes[1, 1]
    ax5.imshow(_render_cleaved_bonds(comp_b))
    ax5.set_title(
        f"5. Closed-Loop Cleaved Bonds (Scene B)\n"
        f"(Regions={comp_b.region_count}, Material={comp_b.material_boundary_count}, "
        f"No Square Seams)",
        fontsize=10.0,
        fontweight="bold",
        pad=8,
    )
    ax5.set_xticks([])
    ax5.set_yticks([])

    # Panel 6: Provenance Recall vs Distractors & Novelty Chart
    ax6 = axes[1, 2]
    comps = [comp_a, comp_b, comp_c]
    labels = ["Scene A\n(Sanctuary)", "Scene B\n(Portrait Mist)", "Scene C\n(Twilight Var)"]
    recalls = [
        float(np.mean([r["self_resonance"] for r in c.atom_provenance_recalls]))
        for c in comps
    ]
    distractors = [
        float(np.mean([r["mean_non_placed_distractor"] for r in c.atom_provenance_recalls]))
        for c in comps
    ]
    novelties = [max(abs(v) for v in c.novelty_vs_sources.values()) for c in comps]

    x = np.arange(len(comps))
    w = 0.25
    b1 = ax6.bar(x - w, recalls, width=w, color="#0284C7", label="Atom Recall")
    b2 = ax6.bar(x, distractors, width=w, color="#94A3B8", label="Non-Placed Distractor")
    b3 = ax6.bar(x + w, novelties, width=w, color="#EA580C", label="Max Copy Corr")
    for bars, col in ((b1, "#0369A1"), (b2, "#475569"), (b3, "#C2410C")):
        for bar in bars:
            val = float(bar.get_height())
            ax6.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.015,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
                color=col,
            )

    ax6.set_xticks(x)
    ax6.set_xticklabels(labels, fontsize=9.0, fontweight="bold")
    ax6.set_ylim(0.0, 1.25)
    ax6.grid(axis="y", alpha=0.25)
    ax6.legend(loc="upper right", fontsize=8.0, framealpha=0.92)
    ax6.set_title(
        f"6. Provenance & Novelty ({len(engine.causal_field.voxels)} Voxels, "
        f"{len(engine.causal_field.beams)} Beams)",
        fontsize=9.8,
        fontweight="bold",
        pad=10,
    )

    fig.suptitle(
        "ELYSIA Topological Mold Composition & Blank-Canvas Recreation (Pure DoG Physics, Zero VLM/LLM)",
        fontsize=13.5,
        fontweight="bold",
        y=0.982,
    )
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.945], h_pad=2.4)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    return output_path


def run_demo() -> Dict[str, Any]:
    t0 = time.perf_counter()
    engine = OcularTopologicalCompositionEngine()

    face_rgb = _load_rgb(ASSETS_DIR / "real_human_face_hd.jpg")
    yose_rgb = _load_rgb(ASSETS_DIR / "genuine_optical_4k_yosemite.jpg")
    leaf_rgb = _load_first_video_frame_rgb(ASSETS_DIR / "real_external_nature_video.mp4")

    face_atoms = engine.harvest_atoms_from_scene(
        face_rgb, scene_name="face_hd", n_fixations=4, fovea_radius_ratio=0.20, atom_size=128
    )
    yose_atoms = engine.harvest_atoms_from_scene(
        yose_rgb, scene_name="yosemite_4k", n_fixations=4, fovea_radius_ratio=0.20, atom_size=128
    )
    leaf_atoms = engine.harvest_atoms_from_scene(
        leaf_rgb, scene_name="macro_leaf", n_fixations=4, fovea_radius_ratio=0.22, atom_size=128
    )

    # Composition A: Botanical Valley Sanctuary (Yosemite peaks + Macro botanical atoms on blank canvas)
    placements_a = [
        AtomPlacement(yose_atoms[2].atom_id, (0.36, 0.28), scale_ratio=0.68, opacity=0.92, chroma_harmonize=0.45),
        AtomPlacement(yose_atoms[3].atom_id, (0.38, 0.72), scale_ratio=0.68, opacity=0.92, chroma_harmonize=0.45),
        AtomPlacement(yose_atoms[0].atom_id, (0.72, 0.30), scale_ratio=0.66, opacity=0.95, chroma_harmonize=0.40),
        AtomPlacement(leaf_atoms[0].atom_id, (0.70, 0.70), scale_ratio=0.66, opacity=0.95, chroma_harmonize=0.35),
        AtomPlacement(leaf_atoms[1].atom_id, (0.78, 0.50), scale_ratio=0.58, opacity=0.90, chroma_harmonize=0.40),
    ]
    comp_a = engine.compose_novel_scene(
        composition_name="botanical_valley_sanctuary",
        placements=placements_a,
        atmosphere_atom_id=yose_atoms[3].atom_id,
        canvas_hw=(240, 320),
        harmonize_atmosphere=0.38,
    )

    # Composition B: Portrait in Alpine Mist (Face structure harmonized with Yosemite mountain mist)
    placements_b = [
        AtomPlacement(yose_atoms[2].atom_id, (0.76, 0.22), scale_ratio=0.64, opacity=0.90, chroma_harmonize=0.30),
        AtomPlacement(yose_atoms[0].atom_id, (0.76, 0.78), scale_ratio=0.64, opacity=0.90, chroma_harmonize=0.30),
        AtomPlacement(face_atoms[0].atom_id, (0.44, 0.50), scale_ratio=0.82, opacity=0.98, chroma_harmonize=0.22),
    ]
    comp_b = engine.compose_novel_scene(
        composition_name="portrait_in_alpine_mist",
        placements=placements_b,
        atmosphere_atom_id=yose_atoms[3].atom_id,
        canvas_hw=(240, 320),
        harmonize_atmosphere=0.24,
    )

    # Composition C: Parametric Variant of Composition A (shifted layout + twilight + +45 deg chroma)
    placements_c = [
        AtomPlacement(yose_atoms[2].atom_id, (0.34, 0.72), scale_ratio=0.72, opacity=0.92, chroma_harmonize=0.45),
        AtomPlacement(yose_atoms[3].atom_id, (0.36, 0.26), scale_ratio=0.66, opacity=0.92, chroma_harmonize=0.45),
        AtomPlacement(leaf_atoms[0].atom_id, (0.68, 0.30), scale_ratio=0.68, opacity=0.95, chroma_harmonize=0.35),
        AtomPlacement(yose_atoms[0].atom_id, (0.72, 0.72), scale_ratio=0.66, opacity=0.95, chroma_harmonize=0.40),
        AtomPlacement(leaf_atoms[1].atom_id, (0.58, 0.50), scale_ratio=0.56, opacity=0.88, chroma_harmonize=0.40),
    ]
    comp_c = engine.compose_novel_scene(
        composition_name="twilight_remodulation_variant",
        placements=placements_c,
        atmosphere_atom_id=yose_atoms[2].atom_id,
        canvas_hw=(240, 320),
        global_luma_shift=-0.40,
        global_chroma_rot_deg=45.0,
        harmonize_atmosphere=0.38,
    )

    sample_atoms = [
        face_atoms[0],
        face_atoms[1],
        yose_atoms[0],
        yose_atoms[3],
        leaf_atoms[0],
        leaf_atoms[1],
    ]
    dash_path = render_dashboard(
        sample_atoms=sample_atoms,
        comp_a=comp_a,
        comp_b=comp_b,
        comp_c=comp_c,
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
        "harvested_atom_count": len(engine.atoms),
        "compositions": [
            _serialize_comp(comp_a),
            _serialize_comp(comp_b),
            _serialize_comp(comp_c),
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
    print("ELYSIA Topological Mold Composition & Blank-Canvas Recreation (Zero VLM/LLM)")
    print("=" * 88)
    print(f"Elapsed Time          : {elapsed_sec:.2f}s | Harvested Atoms = {len(engine.atoms)}")
    for c_info in state["compositions"]:
        print(
            f"  [{c_info['composition_name']:29s}] | regions={c_info['region_count']:2d} | "
            f"max_src_corr={c_info['max_source_persistence_corr']:.3f} | "
            f"atom_recall={c_info['mean_atom_provenance_resonance']:.3f} vs "
            f"distractor={c_info['mean_non_placed_distractor_resonance']:.3f}"
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
