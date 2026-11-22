"""
test_boundary_mirror_self_reconfiguration.py

Verifies ELYSIA's Boundary-as-Mirror Meta-Structural Re-Recognition &
Autopoietic Hierarchy Genesis:
1. Exposes how the initial given flat structure (`LegacyFlatCognitiveStructure`:
   bag-of-stems + 0D scalar axis collapse) fails when facing the External World
   Mirror at the boundary (agent-patient direction reversals and negation/sublation pivots).
2. Verifies that ELYSIA diagnoses her own structural blindness via AST self-inspection
   and correlates the discarded boundary residuals (`접합 꼬리` and pivot markers)
   against external physical force flow.
3. Verifies that she synthesizes, compiles in `SafeASTSandbox`, and hot-swaps a new
   3-Layer Compositional Causal Organ (`analyze_hierarchical_causal_utterance`) +
   Multi-Scale Coupled Axis Ladder (`Micro 2D Field <-> Meso Force Beam <-> Macro Concept`),
   resolving the mirror distortion on unseen holdout episodes.
"""

from __future__ import annotations

from core.sensory.boundary_mirror_self_reconfiguration import (
    BoundaryMirrorSelfReconfigurationEngine,
)


def test_boundary_mirror_exposes_flat_structure_and_discovers_boundary_tails() -> None:
    engine = BoundaryMirrorSelfReconfigurationEngine()
    calib_pairs, _ = engine.build_boundary_mirror_episodes()
    diag = engine.diagnose_internal_structure_via_boundary_mirror(calib_pairs)

    # 1. AST self-inspection identifies `normalize_token` and `extract_content_stems`
    assert len(diag.ast_blind_spots_detected) >= 2
    assert any("normalize_token" in s for s in diag.ast_blind_spots_detected)

    # 2. Legacy flat structure has low internal separation despite high external reality divergence
    assert diag.mean_internal_separation_before < 0.25
    assert diag.mean_external_reality_divergence > 0.65
    assert diag.mirror_distortion_tension > 3.5

    # 3. Correlating discarded boundary residuals against the external mirror discovers source/target tails & pivots
    assert "이" in diag.discovered_source_boundary_tails or "가" in diag.discovered_source_boundary_tails
    assert "을" in diag.discovered_target_boundary_tails or "를" in diag.discovered_target_boundary_tails
    assert "아니라" in diag.discovered_inversion_pivots


def test_autopoietic_hierarchy_recreation_and_unseen_mirror_alignment() -> None:
    engine = BoundaryMirrorSelfReconfigurationEngine()
    report = engine.run_boundary_mirror_self_recreation_cycle()

    # 1. New hierarchical organ is synthesized, AST-validated, and registered in spine
    organ_rec = report["synthesized_organ_record"]
    assert organ_rec.validation_score == 1.0
    assert organ_rec.ast_node_count > 50

    # 2. On unseen holdout pairs, separation jumps from near-zero (<0.20) to strong opposition (>1.20)
    assert report["mean_old_unseen_separation"] < 0.20
    assert report["mean_new_unseen_separation"] > 1.20

    # 3. Internal causal delta aligns with external reality mirror delta (cosine > 0.85)
    assert report["mean_unseen_mirror_alignment_cosine"] > 0.85

    # 4. Multi-Scale Coupled Axis Ladders preserve 2D micro fields and achieve full top-down/bottom-up coherence
    for ladder_id, l_info in report["multiscale_axis_ladders"].items():
        assert l_info["micro_field_shape"] == (16, 16)
        assert l_info["hierarchy_coherence_score"] > 0.98, f"Low ladder coherence on {ladder_id}"
