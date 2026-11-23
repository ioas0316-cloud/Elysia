import math
from pathlib import Path

from core.orchestration.elysia_self_observing_orchestra import (
    ElysiaSelfObservingOrchestra,
)
from core.orchestration.epistemic_harvester_and_language_bridge import (
    CausalLanguagePaletteBridge,
    EpistemicToolHarvester,
)
from core.physics.causal_mechanism_palette import CausalMechanismPalette
from core.physics.spectral_scale_rotor_manifold import (
    ScaleRotorBand,
    SpectralScaleRotorManifold,
)
import numpy as np


def test_deficit_diagnosis_discovery_and_internalization_loop() -> None:
    palette = CausalMechanismPalette.with_default_principles()
    harvester = EpistemicToolHarvester()

    query = "한글 훈민정음 조음 마찰 파동으로 언어를 분석해줘"
    diag_before = harvester.diagnose_deficit(query, palette)
    assert diag_before.has_deficit is True
    assert "hangul" in diag_before.missing_concepts

    candidates = harvester.explore_external_tools(diag_before, max_candidates=5)
    assert len(candidates) >= 1
    cls_names = [c.class_name for c in candidates]
    assert "HangulTopologicalEngine" in cls_names

    top_hangul = next(c for c in candidates if c.class_name == "HangulTopologicalEngine")
    pigment = harvester.internalize_tool_to_palette(top_hangul, palette)
    assert pigment.pigment_id in palette.pigments
    assert pigment.origin == "harvested_external_tool"

    # Re-running deficit check must now hit internalized palette principle!
    diag_after = harvester.diagnose_deficit(query, palette)
    assert diag_after.has_deficit is False
    assert pigment.pigment_id in diag_after.matched_pigments

    # Execute harvested tool directly in O(K) on a SpectralScaleRotorManifold
    sample_m = SpectralScaleRotorManifold(
        dim=1,
        domain_bounds=([0.0], [1.0]),
        bands=[
            ScaleRotorBand(
                level=0,
                dim=1,
                wavevectors=np.array([[2.0 * math.pi * 2.0]]),
                temporal_omegas=np.array([4.0]),
                amplitudes=np.array([1.0 + 0.0j]),
                centers=np.zeros((1, 1)),
                sigmas=np.array([np.inf]),
                is_global=True,
                parent_indices=np.array([-1], dtype=np.int32),
                parent_couplings=np.array([1.0]),
            )
        ],
    )
    out_m = pigment.apply(sample_m, {"text": "까칠한 폭풍"})
    assert math.isfinite(out_m.integrate_over_domain())


def test_hangul_articulatory_wave_rotor_projection() -> None:
    bridge = CausalLanguagePaletteBridge()

    soft_text = "아름다운 노을과 고요한 물결"
    tense_text = "까칠하고 뾰족한 폭풍의 충격"

    m_soft, art_soft = bridge.text_to_rotor_manifold(soft_text)
    m_tense, art_tense = bridge.text_to_rotor_manifold(tense_text)

    assert art_tense["mean_friction"] > art_soft["mean_friction"]
    assert art_tense["mean_tension"] > art_soft["mean_tension"]
    assert art_tense["high_friction_knots"] >= art_soft["high_friction_knots"]

    bal_soft, _ = bridge.infer_balance_and_time(soft_text, art_soft)
    bal_tense, _ = bridge.infer_balance_and_time(tense_text, art_tense)
    assert bal_tense.contrast_tilt > bal_soft.contrast_tilt
    assert bal_tense.scale_tilt > bal_soft.scale_tilt
    assert m_soft.total_modes >= 3 and m_tense.total_modes >= 3


def test_end_to_end_language_to_self_observing_orchestra(tmp_path: Path) -> None:
    slab = tmp_path / "lang_orchestra.slab"
    bridge = CausalLanguagePaletteBridge()

    with ElysiaSelfObservingOrchestra(slab_path=slab, slice_duration_s=0.4) as director:
        r1 = bridge.conduct_from_language(
            "새벽의 부드럽고 평온한 빛과 화음",
            director=director,
            auto_harvest=True,
            manifest=False,
        )
        r2 = bridge.conduct_from_language(
            "황혼의 강렬한 한글 조음 폭풍과 프리즘 대비",
            director=director,
            auto_harvest=True,
            manifest=False,
        )

        assert r1.inferred_time == 0.0
        assert r2.inferred_time == 5.0
        assert r2.deficit_diagnosis.has_deficit is True
        assert r2.harvested_pigment_id is not None
        assert "HangulTopologicalEngine" in r2.harvested_pigment_id

        assert r1.directing_result is not None and r2.directing_result is not None
        ten1 = r1.directing_result.present_slice.performances["narrative.drama_director"].causal_state[
            "dramatic_tension"
        ]
        ten2 = r2.directing_result.present_slice.performances["narrative.drama_director"].causal_state[
            "dramatic_tension"
        ]
        assert ten2 > ten1

        # Honest Chinese-Room disconnection metric must be present and > 0
        disc = r2.tethering_report["experiential_disconnection"]
        assert disc > 0.0

        # Subsequent Hangul query must recall internalized pigment without re-harvesting!
        r3 = bridge.conduct_from_language(
            "한글 조음 파동으로 다시 확인",
            director=director,
            auto_harvest=True,
            manifest=False,
        )
        assert r3.deficit_diagnosis.has_deficit is False
        assert r3.harvested_pigment_id is None
