"""
test_active_ocular_mechanics_engine.py

Integration and physics-verification tests for ActiveOcularMechanicsEngine strictly on
real external 4K/HD photographs and real nature video (zero VLMs, zero LLMs, zero toy circles).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.sensory.active_ocular_mechanics_engine import ActiveOcularMechanicsEngine

ASSETS_DIR = Path(r"c:\Elysia\demos\real_world_assets")
FACE_PATH = ASSETS_DIR / "real_human_face_hd.jpg"
YOSEMITE_PATH = ASSETS_DIR / "genuine_optical_4k_yosemite.jpg"
VIDEO_PATH = ASSETS_DIR / "real_external_nature_video.mp4"


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert bgr is not None, f"Failed to load real asset: {path}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _darken_and_shift(
    full_rgb: np.ndarray,
    box_yxyx: tuple[int, int, int, int],
    brightness_scale: float = 0.55,
    shift_frac: float = 0.05,
) -> np.ndarray:
    h, w = full_rgb.shape[:2]
    y0, x0, y1, x1 = box_yxyx
    bh = max(16, y1 - y0)
    bw = max(16, x1 - x0)
    sy0 = int(np.clip(y0 + round(bh * shift_frac), 0, max(0, h - bh)))
    sx0 = int(np.clip(x0 - round(bw * shift_frac), 0, max(0, w - bw)))
    crop = full_rgb[sy0 : sy0 + bh, sx0 : sx0 + bw].astype(np.float32) / 255.0
    linear = np.power(np.clip(crop, 0.0, 1.0), 2.2) * brightness_scale
    dark = np.power(np.clip(linear, 0.0, 1.0), 1.0 / 2.2)
    return np.clip(np.round(dark * 255.0), 0.0, 255.0).astype(np.uint8)


def test_active_foveal_saccade_and_causal_field_binding() -> None:
    if not FACE_PATH.exists() or not YOSEMITE_PATH.exists():
        pytest.skip("Missing real visual assets")

    face_rgb = _load_rgb(FACE_PATH)
    yosemite_rgb = _load_rgb(YOSEMITE_PATH)

    engine = ActiveOcularMechanicsEngine()
    face_scan = engine.scan_image(
        face_rgb, scene_name="face_hd", n_fixations=4, periphery_max_width=220
    )
    yosemite_scan = engine.scan_image(
        yosemite_rgb, scene_name="yosemite_4k", n_fixations=4, periphery_max_width=220
    )

    assert len(face_scan.foveal_engrams) == 4
    assert len(yosemite_scan.foveal_engrams) == 4

    # IOR prevents gaze from collapsing onto the same fixation point
    for scan in (face_scan, yosemite_scan):
        pts = scan.saccade_path_yx
        dists = [
            float(np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
            for i in range(len(pts))
            for j in range(i + 1, len(pts))
        ]
        min_pair_dist = min(dists)
        min_dim = float(min(scan.image_shape_hw))
        assert min_pair_dist >= 0.05 * min_dim, (
            f"Saccades collapsed too closely ({min_pair_dist:.1f}px vs {min_dim:.1f}px)"
        )
        assert scan.periphery_vs_fovea_detail_gain > 1.0

    # Verify CausalField 6D Voxels and spatial ConnectivityBeams exist
    assert len(engine.causal_field.voxels) == 8
    assert len(engine.causal_field.beams) == 6
    for v in engine.causal_field.voxels.values():
        assert v.position.shape == (6,)
        assert v.tensor.shape == (6,)

    # Photometric darkening (0.55x) + slight shift recognition across both real images
    for img, scan in ((face_rgb, face_scan), (yosemite_rgb, yosemite_scan)):
        target_eng = scan.foveal_engrams[0]
        perturbed = _darken_and_shift(
            img, target_eng.fovea_box_yxyx, brightness_scale=0.55, shift_frac=0.05
        )
        res = engine.recognize_perturbed_patch(
            perturbed, query_name=f"pert_{target_eng.engram_id}", top_k=4
        )
        assert res["best_engram_id"] == target_eng.engram_id, (
            f"Expected {target_eng.engram_id}, got {res['top_matches']}"
        )
        assert res["best_resonance"] >= 0.75


def test_real_video_motion_guided_tracking() -> None:
    if not VIDEO_PATH.exists():
        pytest.skip("Missing real external nature video")

    engine = ActiveOcularMechanicsEngine()
    video_res = engine.observe_video_stream(
        VIDEO_PATH, max_frames=5, frame_stride=8, max_width=200
    )

    assert video_res.frames_processed == 5
    assert len(video_res.frame_events) == 4
    assert len(video_res.motion_gaze_path_yx) == 4
    assert video_res.sample_flow_mag.shape == video_res.sample_moving_mask.shape

    valid_classes = {
        "SELF_EYE_MOTION",
        "STATIC_IDENTICAL",
        "GLOBAL_ILLUMINATION_CHANGE",
        "MOVING_CAST_SHADOW",
        "INDEPENDENT_OBJECT_MOTION",
        "SCENE_CHANGE",
    }
    for ev in video_res.frame_events:
        assert ev["event_classification"] in valid_classes
        assert len(ev["gaze_yx_px"]) == 2
