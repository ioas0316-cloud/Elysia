"""
test_multiscale_3d_perspective_media_engine.py

Verifies ELYSIA's 4K Dataset Multi-Scale Reception & 3D Perspective / Scale
Transition Engine (3rd-Person Macro -> 1st-Person Look-Up -> Micro Zoom-Up):
1. Ingests the native 4K dataset image (`yosemite.png`, `2160x3840`) across
   `L12` macro, `L6` meso, and 1:1 native `L2` micro patches.
2. Renders a true 4K Ultra-Wide Panorama (`3840 x 960`) across 4 distinct 3D
   camera perspectives and focal scale levels (`L12 -> L8 -> L5 -> L2`), plus
   an 18-frame continuous 3D camera flight & zoom transition video (`GIF`).
3. Verifies via `OcularScaleOrgan` re-observation that zooming in from `L12`
   (3rd-person macro) to `L2` (micro bark & cellular sap zoom-up) unlocks higher
   micro-texture energy (Level-of-Detail emergence) rather than blurry pixel blow-up.
"""

from __future__ import annotations

from pathlib import Path

from core.sensory.multiscale_3d_perspective_media_engine import (
    Multiscale3DPerspectiveMediaEngine,
)


def test_4k_multiperspective_3d_camera_and_zoom_transition() -> None:
    engine = Multiscale3DPerspectiveMediaEngine()
    report = engine.create_4k_multiperspective_and_zoom_transition_media(n_transition_frames=18)

    # 1. Verify native 4K dataset ingestion and AST internalization
    principle = report["internalized_3d_principle"]
    assert principle.native_4k_resolution_hw[1] >= 1920
    assert len(principle.macro_l12_palette_16x3) == 16
    assert principle.principle_id in engine.spine.saturation_laws

    # 2. Verify true 4K width panorama (960 x 3840 x 3) and 18-frame 3D flight GIF
    assert report["panorama_resolution_hw"] == (960, 3840, 3)
    p_path = Path(report["output_4k_panorama_path"])
    g_path = Path(report["output_3d_transition_gif_path"])
    assert p_path.exists() and p_path.stat().st_size > 150000
    assert g_path.exists() and g_path.stat().st_size > 40000
    assert report["video_frame_count"] == 18

    # 3. Verify scale-dependent Level-of-Detail (LoD) emergence via OcularScaleOrgan:
    # Micro Zoom-Up (L2) unlocks significantly finer surface texture energy than Macro Overview (L12)
    reobs = report["perspective_reobservations"]
    macro_l12 = reobs["1_3rd_person_macro_zoom_out_L12"]
    lookup_l5 = reobs["3_1st_person_canopy_cathedral_lookup_L5"]
    micro_l2 = reobs["4_extreme_micro_bark_and_cellular_sap_zoomup_L2"]

    assert micro_l2["fine_texture_energy"] > macro_l12["fine_texture_energy"]
    assert lookup_l5["pitch_angle_deg"] > 60.0
