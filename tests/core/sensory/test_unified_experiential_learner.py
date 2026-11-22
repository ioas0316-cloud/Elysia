"""
test_unified_experiential_learner.py

End-to-end verification of ELYSIA's closed World -> Book -> Game -> Creation
experiential learning architecture with ZERO semantic/keyword hardcoding:
1. Spectral purity / white-dilution saturation relational law via self-re-observation
2. Blank-slate axis grounding & confound self-re-observation mitosis (Blind Tokens)
3. Blind Game play via OcularScaleOrgan + MechanoAcousticOrgan (L1 + rotated/dimmed L2)
4. Real Markdown corpus ingestion, Contextual Lexical Mitosis, Stem/Branch comparison & 100% unseen disambiguation (Korean + Blind)
5. EpistemicVoid lifecycle & empirical Book-to-Game causal rule verification
6. Autonomous Creation of novel Level 3 puzzle & multi-axis visual synthesis, self-verified through own organs
7. JSON cold-reload persistence roundtrip & CausalField multi-scale L6..L14 projection
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.sensory.epistemic_knowledge_spine import EpistemicKnowledgeSpine
from core.sensory.imaginative_perceptual_recreator import calibrate_perceptual_manifold
from core.sensory.ocular_scale_organ import OcularScaleOrgan
from core.sensory.unified_experiential_learner import (
    CORPUS_DIR,
    BookReadingLearner,
    UnifiedSessionReport,
    WorldPerceptualLearner,
    _extract_tile_optical_signature,
    _load_real_yosemite_scene,
    build_curriculum_bundle,
    normalize_token,
    run_unified_world_game_book_session,
)
from simulators.causal_exploration_game import CausalExplorationGame


@pytest.fixture(scope="module")
def korean_report() -> UnifiedSessionReport:
    return run_unified_world_game_book_session(use_blind_tokens=False)


@pytest.fixture(scope="module")
def blind_report() -> UnifiedSessionReport:
    return run_unified_world_game_book_session(use_blind_tokens=True)


def test_saturation_spectral_purity_reobservation_law(korean_report: UnifiedSessionReport) -> None:
    law = korean_report.saturation_law
    assert law.monotone_chroma_decay is True
    assert law.chroma_radii[0] > 1.8
    assert law.chroma_radii[-1] < 0.05
    assert law.max_hue_drift_deg_core < 25.0
    assert law.ortho_ev_chroma_drift < 0.08
    assert law.ortho_rough_l2_min >= 0.85
    assert law.boundary_tipping_w is not None
    assert 0.4 <= law.boundary_tipping_w <= 1.0


def test_blank_slate_axis_grounding_and_confound_mitosis_blind_tokens(
    korean_report: UnifiedSessionReport,
    blind_report: UnifiedSessionReport,
) -> None:
    expected_axes = {
        "chroma_high": "chroma_radius",
        "luma_high": "luminance_a",
        "rough_high": "fine_roughness",
        "stiff_high": "stiffness_log10",
        "ring_high": "acoustic_freq_log10",
    }
    for rep in (korean_report, blind_report):
        for key, ax_name in expected_axes.items():
            binding = rep.axis_bindings[key]
            assert binding.axis_name == ax_name
            assert binding.direction == pytest.approx(1.0)
            assert binding.reobservation_verified is True
            assert binding.reobservation_margin > 0.15

        disamb_revs = [r for r in rep.spine.criteria_revisions if r["kind"] == "AXIS_DISAMBIGUATION"]
        assert len(disamb_revs) >= 2
        assert len(rep.spine.get_open_voids("VOID_AMBIGUOUS_AXIS")) == 0

    # Verify stage-by-stage that ambiguous confounds first open VOID_AMBIGUOUS_AXIS before self-re-observation resolves them
    fresh_spine = EpistemicKnowledgeSpine()
    fresh_world = WorldPerceptualLearner(fresh_spine)
    scene = _load_real_yosemite_scene()
    curriculum = build_curriculum_bundle(use_blind_tokens=True)
    fresh_world.ground_and_disambiguate_perceptual_symbols(
        scene_rgb=scene,
        symbol_map=curriculum["symbol_map"],
        oracle_fn=curriculum["oracle_fn"],
    )
    resolved_amb = [
        v for v in fresh_spine.voids if v.kind == "VOID_AMBIGUOUS_AXIS" and v.status == "RESOLVED"
    ]
    assert len(resolved_amb) >= 2


def test_blind_game_playing_cleavage_and_level2_transfer(
    korean_report: UnifiedSessionReport,
    blind_report: UnifiedSessionReport,
) -> None:
    for rep in (korean_report, blind_report):
        g1 = rep.game_level1
        g2 = rep.game_level2
        assert g1.cleared is True
        assert g1.hazard_hits <= 1
        assert g1.cleaved_barriers >= 1
        assert g1.cleared_foams >= 1

        assert g2.cleared is True
        assert g2.hazard_hits == 0
        assert g2.cleaved_barriers >= 1
        assert g2.cleared_foams >= 1

        concept_splits = [e for e in rep.spine.mitosis_events if e["kind"] == "CONCEPT_MITOSIS"]
        assert len(concept_splits) >= 2

    # Directly verify Timber grain rotation between Level 1 (0 deg) and Level 2 (90 deg) through OcularScaleOrgan
    eye = OcularScaleOrgan()
    man = calibrate_perceptual_manifold(eye, _load_real_yosemite_scene())
    g1_env = CausalExplorationGame(level_id=1)
    g2_env = CausalExplorationGame(level_id=2)
    _, sig1 = _extract_tile_optical_signature(eye, man, g1_env.foveate_tile(1, 2))
    _, sig2 = _extract_tile_optical_signature(eye, man, g2_env.foveate_tile(2, 1))
    diff = abs(sig1["phi_opt_deg"] - sig2["phi_opt_deg"]) % 180.0
    assert abs(diff - 90.0) < 12.0


def test_real_corpus_and_contextual_polysemy_mitosis_korean_and_blind(
    korean_report: UnifiedSessionReport,
    blind_report: UnifiedSessionReport,
) -> None:
    # Morphological stemmer invariants (without semantic word lists)
    assert normalize_token("명도") == "명도"
    assert normalize_token("채도") == "채도"
    assert normalize_token("단단함") == "단단하다"
    assert normalize_token("밝은") == "밝다"
    assert normalize_token("사과를") == "사과"
    assert normalize_token("손으로") == "손"

    expected_corpus_n = len(list(CORPUS_DIR.glob("*.md")))
    assert korean_report.corpus_files_read == expected_corpus_n >= 5

    for rep in (korean_report, blind_report):
        assert rep.disambiguation_accuracy == pytest.approx(1.0)
        assert len(rep.disambiguation_details) == 6
        assert all(d["correct"] and d["margin"] > 0.01 for d in rep.disambiguation_details)

        for _, raw_form in rep.polysemy_targets.items():
            form = normalize_token(raw_form)
            senses = rep.spine.symbols[form]
            assert len(senses) >= 2
            comps = rep.sense_comparisons[form]
            assert len(comps) >= 1
            for comp in comps:
                assert comp.structural_stem_similarity >= 0.70
                assert comp.context_divergence >= 0.70
                assert len(comp.shared_stem_dimensions) >= 1
                assert len(comp.divergent_branch_dimensions) >= 1


def test_void_detection_autonomous_filling_and_book_to_game_verification(
    korean_report: UnifiedSessionReport,
    blind_report: UnifiedSessionReport,
) -> None:
    for rep in (korean_report, blind_report):
        timeline = {row["stage"]: row for row in rep.void_timeline}
        assert timeline["2_Book_Read"]["open_voids"] >= 5
        assert timeline["4_Game_L2"]["resolved_voids"] > timeline["2_Book_Read"]["resolved_voids"]
        assert timeline["6_Verify"]["resolved_voids"] >= 12

        assert len(rep.spine.causal_rules) == 3
        for rule in rep.spine.causal_rules.values():
            assert rule.source_medium == "book"
            assert rule.verified_in_reality is True
            assert rule.confirmations >= 1

        # Genuine ungrounded abstract corpus symbols remain honestly marked open
        ungrounded_open = rep.spine.get_open_voids("VOID_UNGROUNDED_SYMBOL")
        assert len(ungrounded_open) >= 3


def test_autonomous_grounded_creation_and_self_verification(
    korean_report: UnifiedSessionReport,
    blind_report: UnifiedSessionReport,
) -> None:
    for rep in (korean_report, blind_report):
        cl3 = rep.created_level3
        assert cl3.verified is True
        assert cl3.play_summary.cleared is True
        assert cl3.play_summary.hazard_hits == 0
        assert cl3.play_summary.cleaved_barriers >= 1
        assert cl3.reobserved_hazard_chroma > cl3.reobserved_sunlit_chroma + 0.25
        assert cl3.reobserved_timber_coherence > 0.62

        cvis = rep.created_visual
        assert cvis.verified is True
        assert cvis.target_axes == ["chroma_radius", "fine_roughness"]
        assert cvis.created_metrics["chroma_radius"] > 0.10
        assert cvis.created_metrics["fine_roughness"] > 0.15
        assert abs(cvis.created_metrics["luminance_a"]) < 0.08
        assert cvis.l2_persistence_corr >= 0.90

        assert len(rep.spine.creations) == 2
        assert all(c.verified_by_own_organs for c in rep.spine.creations)


def test_persistence_roundtrip_and_causal_field_projection(
    korean_report: UnifiedSessionReport, tmp_path: Path
) -> None:
    save_path = tmp_path / "elysia_epistemic_spine.json"
    korean_report.spine.save_json(save_path)
    assert save_path.exists()

    reloaded = EpistemicKnowledgeSpine.load_json(save_path)
    assert len(reloaded.concepts) == len(korean_report.spine.concepts)
    assert set(reloaded.symbols.keys()) == set(korean_report.spine.symbols.keys())
    for sym, senses in korean_report.spine.symbols.items():
        assert len(reloaded.symbols[sym]) == len(senses)
    assert len(reloaded.causal_rules) == len(korean_report.spine.causal_rules)
    assert all(r.verified_in_reality for r in reloaded.causal_rules.values())
    assert len(reloaded.voids) == len(korean_report.spine.voids)
    assert len(reloaded.mitosis_events) == len(korean_report.spine.mitosis_events)
    assert len(reloaded.creations) == len(korean_report.spine.creations)
    assert reloaded.saturation_laws == korean_report.spine.saturation_laws

    # Verify reloaded axis binding & unseen book disambiguation without re-training
    assert reloaded.get_axis_binding("채도") is not None
    assert reloaded.get_axis_binding("채도").axis_name == "chroma_radius"  # type: ignore[union-attr]
    curriculum = build_curriculum_bundle(use_blind_tokens=False)
    acc_reload, _ = BookReadingLearner(reloaded).evaluate_disambiguation_benchmark(
        curriculum["unseen_tests"]
    )
    assert acc_reload == pytest.approx(1.0)

    # Verify CausalField L6..L14 multi-scale topology & positive semantic mass
    cf = korean_report.causal_field
    scales = {int(round(v.scale_level)) for v in cf.voxels.values()}
    assert {6, 10, 12, 14}.issubset(scales)
    assert len(cf.beams) >= 10
    assert all(v.mass > 0.0 for v in cf.voxels.values())
