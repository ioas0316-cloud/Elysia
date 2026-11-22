"""
test_autopoietic_visual_media_creator.py

Verifies ELYSIA's Autonomous Dataset Principle Extraction, Self-Code Spectrum
Internalization, Circular Painting Self-Revision, and Living Video (`GIF`) Creation:
1. Formulates top-level sovereign creative intent (`What`, `Why`, `Purpose`) before
   invoking lower-level numerical/brush tools.
2. Observes real landscape dataset imagery (`yosemite.png`) AND her own rendering
   AST code to internalize a 16-stratum atmospheric palette, 3D relief lighting vector,
   and multi-scale brush grain spectrum into persistent spine memory.
3. Paints her World-Tree landscape across 3 self-reobserved revision stages
   (`Draft 0 -> Draft 1 -> Final Masterpiece`), strictly increasing fine texture richness
   and chromatic depth as verified by `OcularScaleOrgan`.
4. Synthesizes a 16-frame continuous phase-animated video (`GIF`) and recalls the
   internalized spectrum from memory to paint a new mood without re-coding wavelength laws.
"""

from __future__ import annotations

from pathlib import Path

from core.sensory.autopoietic_visual_media_creator import AutopoieticVisualMediaCreator


def test_dataset_and_self_ast_spectrum_internalization_and_painting_video_creation() -> None:
    creator = AutopoieticVisualMediaCreator()
    report = creator.create_self_reobserved_painting_and_video(n_video_frames=16)

    # 1. Sovereign intent precedes execution
    intent = report["sovereign_intent"]
    assert "세계수" in intent.what_to_create
    assert len(intent.required_scale_strata) == 4

    # 2. Spectrum principle is extracted from both dataset:yosemite.png and self-AST and persisted in spine
    spec = report["internalized_spectrum"]
    assert "dataset:yosemite.png" in spec.origin_dataset_or_ast_func
    assert len(spec.vertical_sky_to_earth_palette_srgb) == 16
    assert len(spec.ast_primitive_signature) >= 3
    assert spec.principle_id in creator.spine.saturation_laws

    # 3. Circular painting self-revision (Draft 0 -> Draft 1 -> Final) monotonically enriches texture & depth
    revs = report["painting_revisions"]
    assert len(revs) == 3
    assert revs[2].fine_texture_richness > revs[0].fine_texture_richness * 1.08
    assert revs[2].ocular_region_count >= revs[0].ocular_region_count * 2
    assert revs[2].ocular_material_boundaries >= revs[0].ocular_material_boundaries * 2

    # 4. Both the 4-panel progression PNG and the 16-frame animated GIF exist on disk and show smooth temporal motion
    png_path = Path(report["output_painting_path"])
    gif_path = Path(report["output_video_gif_path"])
    assert png_path.exists() and png_path.stat().st_size > 5000
    assert gif_path.exists() and gif_path.stat().st_size > 15000
    assert report["video_frame_count"] == 16
    assert report["mean_temporal_frame_motion"] > 1.0
    assert report["temporal_smoothness"] > 0.35
