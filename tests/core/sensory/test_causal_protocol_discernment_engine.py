"""
test_causal_protocol_discernment_engine.py

Physics-grounded verification of ontological protocol discernment across physical photos,
macro video frames, chromatically modulated hybrid reality, cel-shaded anime graphics,
and vector/UI dashboards (zero filename hints, zero VLMs, zero LLMs).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.sensory.causal_protocol_discernment_engine import (
    CausalProtocolDiscernmentEngine,
    stylize_to_cel_anime_digital,
)

ASSETS_DIR = Path(r"c:\Elysia\demos\real_world_assets")
FACE_PATH = ASSETS_DIR / "real_human_face_hd.jpg"
YOSEMITE_PATH = ASSETS_DIR / "genuine_optical_4k_yosemite.jpg"
VIDEO_PATH = ASSETS_DIR / "real_external_nature_video.mp4"
UI_PATH = Path(r"c:\Elysia\docs\assets\elysia_unified_mind_organism_dashboard.png")


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


def test_protocol_discernment_and_scale_meeting_points() -> None:
    if (
        not FACE_PATH.exists()
        or not YOSEMITE_PATH.exists()
        or not VIDEO_PATH.exists()
        or not UI_PATH.exists()
    ):
        pytest.skip("Missing required real/digital assets")

    face_rgb = _load_rgb(FACE_PATH)
    yose_rgb = _load_rgb(YOSEMITE_PATH)
    macro_rgb = _load_first_video_frame_rgb(VIDEO_PATH)
    ui_rgb = _load_rgb(UI_PATH)

    cel_face_rgb = stylize_to_cel_anime_digital(face_rgb, n_colors=10, max_width=480)
    cel_sky_rgb = stylize_to_cel_anime_digital(yose_rgb, n_colors=10, max_width=480)

    engine = CausalProtocolDiscernmentEngine()
    yose_fact = engine.gen.factorize(yose_rgb, source_name="yose_src_t", max_width=320)
    mod_sky_rgb = engine.gen.modulate(
        yose_fact, name="mod_sky_t", chroma_rotation_deg=110.0, chroma_gain=1.18
    ).synthesized_rgb

    r_face = engine.discern_protocol(face_rgb, "real_face_photo", max_width=240)
    r_yose = engine.discern_protocol(yose_rgb, "real_yosemite_sky", max_width=240)
    r_macro = engine.discern_protocol(macro_rgb, "real_macro_video", max_width=240)
    r_mod = engine.discern_protocol(mod_sky_rgb, "modulated_neon_sky", max_width=240)
    r_cel_face = engine.discern_protocol(cel_face_rgb, "anime_cel_face", max_width=240)
    r_cel_sky = engine.discern_protocol(cel_sky_rgb, "anime_cel_sky", max_width=240)
    r_ui = engine.discern_protocol(ui_rgb, "digital_ui_dashboard", max_width=240)

    # Print full diagnostic metrics for empirical visibility
    for rep in (r_face, r_yose, r_macro, r_mod, r_cel_face, r_cel_sky, r_ui):
        o = rep.observables
        print(
            f"[{rep.stimulus_name:20s}] {rep.protocol_verdict:27s} | "
            f"P={rep.physical_reality_score:.2f} H={rep.hybrid_modulation_score:.2f} D={rep.digital_realization_score:.2f} | "
            f"flat={o.flat_plateau_ratio:.2f} pal={o.palette_top12_concentration:.2f} "
            f"rect={o.rectilinear_axis_ratio:.2f} bim={o.edge_bimodality_index:.2f} chr={o.chromatic_manifold_deviation:.2f}"
        )

    # 1. Three physical real-world optical stimuli
    assert r_face.protocol_verdict == "PHYSICAL_OPTICAL_REALITY"
    assert r_yose.protocol_verdict == "PHYSICAL_OPTICAL_REALITY"
    assert r_macro.protocol_verdict == "PHYSICAL_OPTICAL_REALITY"

    # 2. Three digital realizations (anime cel face, anime cel sky, digital UI dashboard)
    assert r_cel_face.protocol_verdict == "DIGITAL_SYNTHETIC_REALIZATION"
    assert r_cel_sky.protocol_verdict == "DIGITAL_SYNTHETIC_REALIZATION"
    assert r_ui.protocol_verdict == "DIGITAL_SYNTHETIC_REALIZATION"

    # 3. Hybrid chromatically modulated reality
    assert r_mod.protocol_verdict == "HYBRID_MODULATED_REALITY"

    # 4. Meeting & divergence comparison across scale
    cmp_face = engine.compare_protocol_meeting(r_face, r_cel_face)
    cmp_sky_ui = engine.compare_protocol_meeting(r_yose, r_ui)

    assert cmp_face.macro_meeting_score >= 0.60
    assert cmp_face.micro_substrate_divergence >= 0.45
    assert cmp_sky_ui.macro_meeting_score < cmp_face.macro_meeting_score

    # 5. CausalField 6D Voxels & ConnectivityBeams
    for rep in (r_face, r_yose, r_macro, r_mod, r_cel_face, r_cel_sky, r_ui):
        assert rep.voxel_id in engine.causal_field.voxels
        v = engine.causal_field.voxels[rep.voxel_id]
        assert v.position.shape == (6,)
        assert v.tensor.shape == (6,)
    assert len(engine.causal_field.beams) >= 2
