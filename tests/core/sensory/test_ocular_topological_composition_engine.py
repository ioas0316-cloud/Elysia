"""
test_ocular_topological_composition_engine.py

Physics & provenance verification tests for OcularTopologicalCompositionEngine
on real 4K photographs and nature video frame (zero VLM/LLM/diffusion).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.sensory.ocular_topological_composition_engine import (
    AtomPlacement,
    OcularTopologicalCompositionEngine,
)

ASSETS_DIR = Path(r"c:\Elysia\demos\real_world_assets")
FACE_PATH = ASSETS_DIR / "real_human_face_hd.jpg"
YOSEMITE_PATH = ASSETS_DIR / "genuine_optical_4k_yosemite.jpg"
VIDEO_PATH = ASSETS_DIR / "real_external_nature_video.mp4"


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert bgr is not None, f"Failed to load image: {path}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_first_video_frame_rgb(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"Failed to open video: {path}"
    try:
        ok, frame_bgr = cap.read()
    finally:
        cap.release()
    assert ok and frame_bgr is not None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def test_harvest_blank_canvas_composition_novelty_and_provenance() -> None:
    if not FACE_PATH.exists() or not YOSEMITE_PATH.exists() or not VIDEO_PATH.exists():
        pytest.skip("Missing real visual assets")

    face_rgb = _load_rgb(FACE_PATH)
    yose_rgb = _load_rgb(YOSEMITE_PATH)
    leaf_rgb = _load_first_video_frame_rgb(VIDEO_PATH)

    engine = OcularTopologicalCompositionEngine()
    face_atoms = engine.harvest_atoms_from_scene(
        face_rgb, scene_name="face_hd", n_fixations=3, fovea_radius_ratio=0.20, atom_size=128
    )
    yose_atoms = engine.harvest_atoms_from_scene(
        yose_rgb, scene_name="yosemite_4k", n_fixations=3, fovea_radius_ratio=0.20, atom_size=128
    )
    leaf_atoms = engine.harvest_atoms_from_scene(
        leaf_rgb, scene_name="macro_leaf", n_fixations=3, fovea_radius_ratio=0.22, atom_size=128
    )

    assert len(engine.atoms) >= 9

    # Organic support masks must fade smoothly to < 0.05 along all four rectangular borders
    for atom in engine.atoms.values():
        m = atom.organic_mask
        assert float(np.max(m[0, :])) < 0.05
        assert float(np.max(m[-1, :])) < 0.05
        assert float(np.max(m[:, 0])) < 0.05
        assert float(np.max(m[:, -1])) < 0.05
        assert float(np.max(m)) > 0.50

    placements = [
        AtomPlacement(yose_atoms[1].atom_id, (0.36, 0.28), scale_ratio=0.68, opacity=0.92, chroma_harmonize=0.45),
        AtomPlacement(yose_atoms[0].atom_id, (0.72, 0.30), scale_ratio=0.66, opacity=0.95, chroma_harmonize=0.40),
        AtomPlacement(face_atoms[0].atom_id, (0.46, 0.54), scale_ratio=0.78, opacity=0.96, chroma_harmonize=0.35),
        AtomPlacement(leaf_atoms[0].atom_id, (0.72, 0.72), scale_ratio=0.66, opacity=0.95, chroma_harmonize=0.35),
    ]
    res = engine.compose_novel_scene(
        composition_name="cross_domain_novel_test",
        placements=placements,
        atmosphere_atom_id=yose_atoms[2].atom_id,
        canvas_hw=(240, 320),
        harmonize_atmosphere=0.35,
    )

    # Dynamic range and genuine canvas shape
    assert res.synthesized_rgb.shape == (240, 320, 3)
    assert float(np.std(res.synthesized_rgb.astype(np.float32))) > 15.0

    # Genuine novelty: NOT copying any full original photo
    max_src_corr = max(abs(v) for v in res.novelty_vs_sources.values())
    assert max_src_corr < 0.55, f"Composed scene too correlated with a source: {res.novelty_vs_sources}"

    # Optical coherence & non-trivial cleaved structure
    assert res.region_count >= 2
    assert res.cleaved_bond_ratio > 0.03

    # Component provenance recall inside the composed scene
    mean_self_recall = float(np.mean([r["self_resonance"] for r in res.atom_provenance_recalls]))
    mean_distractor = float(np.mean([r["mean_non_placed_distractor"] for r in res.atom_provenance_recalls]))
    assert mean_self_recall >= 0.65
    assert mean_self_recall > mean_distractor

    # CausalField 6D voxel + connectivity beams to constituent atoms
    assert res.voxel_id in engine.causal_field.voxels
    comp_v = engine.causal_field.voxels[res.voxel_id]
    assert comp_v.position.shape == (6,)
    assert comp_v.tensor.shape == (6,)
    connected_parents = {
        b.source_id for b in engine.causal_field.beams if b.target_id == res.voxel_id
    }
    for pl in placements:
        assert engine.atoms[pl.atom_id].voxel_id in connected_parents
