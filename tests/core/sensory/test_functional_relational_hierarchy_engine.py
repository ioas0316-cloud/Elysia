"""
test_functional_relational_hierarchy_engine.py

Verifies 4-Tier Functional-Relational Stratified Discernment (Sensory -> Functional Organs
bound to Language -> Entity/Agency -> Media Protocol) strictly without VLMs, LLMs, or hints.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.sensory.causal_protocol_discernment_engine import stylize_to_cel_anime_digital
from core.sensory.functional_relational_hierarchy_engine import (
    BehavioralAgencyTrace,
    FunctionalRelationalHierarchyEngine,
    render_minimalist_cartoon_animal,
    render_stone_effigy_from_photo,
)

ASSETS_DIR = Path(r"c:\Elysia\demos\real_world_assets")
FACE_PATH = ASSETS_DIR / "real_human_face_hd.jpg"
YOSEMITE_PATH = ASSETS_DIR / "genuine_optical_4k_yosemite.jpg"


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert bgr is not None, f"Failed to load image: {path}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def test_stratified_organs_agency_and_protocol_vs_flat_pixels() -> None:
    if not FACE_PATH.exists() or not YOSEMITE_PATH.exists():
        pytest.skip("Missing required real assets")

    face_rgb = _load_rgb(FACE_PATH)
    yose_rgb = _load_rgb(YOSEMITE_PATH)
    cel_face_rgb = stylize_to_cel_anime_digital(face_rgb, n_colors=10, max_width=480)
    dot_animal_rgb = render_minimalist_cartoon_animal((280, 280))
    stone_statue_rgb = render_stone_effigy_from_photo(face_rgb)

    engine = FunctionalRelationalHierarchyEngine()

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

    for j in (j_real, j_cel, j_dot, j_stone, j_yose):
        orgs = [o.organ_type for o in j.detected_organs]
        print(
            f"[{j.stimulus_name:22s}] entity={j.entity_verdict:24s} proto={j.protocol_report.protocol_verdict:27s} "
            f"triad={j.organ_triad_coherence:.3f} pix={j.pixel_similarity_to_reference} iso={j.functional_isomorphism_to_reference} orgs={orgs}"
        )

    # 1. Real Human: LIVING_ANIMAL_AGENT + PHYSICAL_OPTICAL_REALITY + EYE_PAIR & MOUTH
    real_org_types = {o.organ_type for o in j_real.detected_organs}
    assert j_real.entity_verdict == "LIVING_ANIMAL_AGENT"
    assert j_real.protocol_report.protocol_verdict == "PHYSICAL_OPTICAL_REALITY"
    assert {"EYE_PAIR", "MOUTH"}.issubset(real_org_types)

    # 2. Cel-Anime Human: CHARACTER_REPRESENTATION + DIGITAL_SYNTHETIC_REALIZATION
    assert j_cel.entity_verdict == "CHARACTER_REPRESENTATION"
    assert j_cel.protocol_report.protocol_verdict == "DIGITAL_SYNTHETIC_REALIZATION"

    # 3. Minimalist 2-Dot Cartoon Animal: low pixel similarity yet high functional isomorphism!
    dot_org_types = {o.organ_type for o in j_dot.detected_organs}
    assert j_dot.entity_verdict == "CHARACTER_REPRESENTATION"
    assert j_dot.protocol_report.protocol_verdict == "DIGITAL_SYNTHETIC_REALIZATION"
    assert {"EYE_PAIR", "MOUTH"}.issubset(dot_org_types)
    assert j_dot.pixel_similarity_to_reference is not None and j_dot.pixel_similarity_to_reference < 0.45
    assert j_dot.functional_isomorphism_to_reference is not None and j_dot.functional_isomorphism_to_reference >= 0.70

    # 4. Lifeless Stone Statue: high pixel similarity yet INANIMATE_EFFIGY (zero living agency!)
    assert j_stone.pixel_similarity_to_reference is not None and j_stone.pixel_similarity_to_reference > 0.65
    assert j_stone.entity_verdict == "INANIMATE_EFFIGY"
    assert j_stone.protocol_report.protocol_verdict == "PHYSICAL_OPTICAL_REALITY"

    # 5. Yosemite Landscape: NON_AGENT_ENVIRONMENT + PHYSICAL_OPTICAL_REALITY
    assert j_yose.entity_verdict == "NON_AGENT_ENVIRONMENT"
    assert j_yose.protocol_report.protocol_verdict == "PHYSICAL_OPTICAL_REALITY"

    # 6. CausalField 6D Hierarchical Voxel + Beam graph
    for j in (j_real, j_cel, j_dot, j_stone, j_yose):
        assert j.entity_voxel_id in engine.causal_field.voxels
        parents = {
            b.source_id for b in engine.causal_field.beams if b.target_id == j.entity_voxel_id
        }
        assert j.protocol_report.voxel_id in parents
        for o in j.detected_organs:
            assert o.voxel_id in parents
