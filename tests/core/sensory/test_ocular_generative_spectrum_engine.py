"""
test_ocular_generative_spectrum_engine.py

Physics & closed-loop self-verification tests for OcularGenerativeSpectrumEngine
using real photographs and nature video frame (zero VLM/LLM/diffusion).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.sensory.ocular_generative_spectrum_engine import OcularGenerativeSpectrumEngine

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


def test_factorization_reproduction_modulation_and_cross_scene_recreation() -> None:
    if not FACE_PATH.exists() or not YOSEMITE_PATH.exists() or not VIDEO_PATH.exists():
        pytest.skip("Missing real visual assets")

    face_rgb = _load_rgb(FACE_PATH)
    yose_rgb = _load_rgb(YOSEMITE_PATH)
    leaf_rgb = _load_first_video_frame_rgb(VIDEO_PATH)

    engine = OcularGenerativeSpectrumEngine()
    face_fact = engine.factorize(face_rgb, source_name="face_hd", max_width=220)
    yose_fact = engine.factorize(yose_rgb, source_name="yosemite_4k", max_width=220)
    leaf_fact = engine.factorize(leaf_rgb, source_name="macro_leaf", max_width=220)

    # 1. Faithful Reproduction
    face_recon = engine.reconstruct(face_fact)
    yose_recon = engine.reconstruct(yose_fact)
    for recon in (face_recon, yose_recon):
        assert recon.psnr_db is not None and recon.psnr_db >= 30.0
        assert recon.structural_corr_with_source >= 0.95
        assert recon.optical_resonance_with_source >= 0.92

    # 2. Orthogonal Illumination Axis Modulation (light changes cleanly, form & chroma stay invariant)
    yose_dark = engine.modulate(
        yose_fact, name="yose_dark_test", luma_shift=-0.55
    )
    assert abs(yose_dark.delta_luma_mean) >= 0.25
    assert yose_dark.structural_corr_with_source >= 0.85
    assert yose_dark.delta_chroma_mean <= 0.08

    # 3. Orthogonal Chromatic Opponent Axis Modulation (hue rotates cleanly, form stays intact)
    yose_rot = engine.modulate(
        yose_fact, name="yose_rot110_test", chroma_rotation_deg=110.0, chroma_gain=1.15
    )
    assert yose_rot.delta_chroma_mean >= 0.12
    assert yose_rot.structural_corr_with_source >= 0.80

    # 4. Multi-Scale Spectrum Dial (painterly suppresses fine bands, relief amplifies fine bands)
    face_paint = engine.modulate(
        face_fact,
        name="face_paint_test",
        band_gains=(0.05, 0.15, 0.70, 1.15, 1.15),
    )
    face_relief = engine.modulate(
        face_fact,
        name="face_relief_test",
        band_gains=(2.2, 1.8, 1.1, 1.0, 1.0),
    )
    orig_fine = float(
        face_fact.scale_band_energies[0] + face_fact.scale_band_energies[1]
    )
    paint_fine = float(
        face_paint.reobserved_scale_profile[0] + face_paint.reobserved_scale_profile[1]
    )
    relief_fine = float(
        face_relief.reobserved_scale_profile[0] + face_relief.reobserved_scale_profile[1]
    )
    assert paint_fine < orig_fine
    assert relief_fine > orig_fine

    # 5. Cross-Scene Recreation (Yosemite boundary mold x Macro Leaf chromatic/scale spectrum)
    rec = engine.recreate_cross_scene(
        structure_fact=yose_fact,
        palette_fact=leaf_fact,
        name="yose_mold_x_leaf_palette_test",
    )
    assert rec.structural_corr_with_source >= 0.72
    dist_before = float(
        np.linalg.norm(yose_fact.chromatic_signature - leaf_fact.chromatic_signature)
    )
    dist_after = float(
        np.linalg.norm(rec.reobserved_chromatic_signature - leaf_fact.chromatic_signature)
    )
    assert dist_after < dist_before, (
        f"Expected recreated chromatic signature closer to donor: {dist_after:.3f} vs {dist_before:.3f}"
    )

    # 6. CausalField 6D Voxels & ConnectivityBeams verification
    expected_voxels = {
        face_fact.voxel_id,
        yose_fact.voxel_id,
        leaf_fact.voxel_id,
        face_recon.voxel_id,
        yose_recon.voxel_id,
        yose_dark.voxel_id,
        yose_rot.voxel_id,
        face_paint.voxel_id,
        face_relief.voxel_id,
        rec.voxel_id,
    }
    assert expected_voxels.issubset(set(engine.causal_field.voxels.keys()))
    assert len(engine.causal_field.beams) >= 8
    for vid in expected_voxels:
        v = engine.causal_field.voxels[vid]
        assert v.position.shape == (6,)
        assert v.tensor.shape == (6,)
