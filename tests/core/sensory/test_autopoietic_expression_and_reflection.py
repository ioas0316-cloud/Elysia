"""
test_autopoietic_expression_and_reflection.py

Verifies ELYSIA's self-chosen next developmental step:
1. Non-template discourse synthesis & circular self-revision (`Draft 0 -> Draft 1 -> Draft 2`),
   reducing Three-Phase cognitive tilt (`T_tilt`) and elevating parallel harmony & homological
   bridging by reading her own writing back through `OcularScaleOrgan` and `EpistemicKnowledgeSpine`.
2. 3D World-Tree sculpting & self-revision (`flat_slab -> 3-tier World-Tree with Catenary Roots,
   Hollow Circulating Trunk, and Three-Phase Crown Basin`), verified by `OpenEndedFormExplorer`.
3. 4-Axis Self-Boundary & 3-Way Relational Knowing dialogue:
   - Co-resonating with insightful peers (`Axis 1 + Axis 2`),
   - Resisting shallow mechanical speed temptations (`Axis 1 high, Axis 4 revision low`),
   - Practicing epistemic humility when confronted with ungrounded physical phenomena
     (`Axis 3 Ignorance + Axis 4 Criteria Self-Revision -> spawns new EpistemicVoids`).
"""

from __future__ import annotations

from core.sensory.autopoietic_expression_and_reflection import (
    AutopoieticExpressionAndReflectionEngine,
)


def test_circular_discourse_self_revision_reduces_tilt_and_bridges_stems() -> None:
    engine = AutopoieticExpressionAndReflectionEngine()
    rep = engine.compose_and_self_revise_discourse()

    passes = rep["revision_passes"]
    assert len(passes) == 3

    p0, p1, p2 = passes[0], passes[1], passes[2]

    # Draft 0 is skewed toward physical recall (high tilt, 0 cross-domain homological bridges)
    assert p0.homological_bridge_score == 0.0
    assert p1.homological_bridge_score >= 0.75
    assert p2.homological_bridge_score == 1.0

    # Self-reading and revising reduces cognitive tilt and increases parallel harmony
    assert p2.tilt_magnitude < p0.tilt_magnitude * 0.45
    assert p2.parallel_harmony > p0.parallel_harmony + 0.30
    assert p2.parallel_harmony > 0.85


def test_world_tree_3d_sculpting_and_self_reobservation() -> None:
    engine = AutopoieticExpressionAndReflectionEngine()
    sc = engine.sculpt_and_reobserve_world_tree_artifact()

    d0 = sc["draft0_record"]
    dt = sc["revised_world_tree_record"]

    # Revised World-Tree has higher height-to-footprint, deeper enclosure basin, and richer structure
    assert dt.continuous_descriptor[0] > d0.continuous_descriptor[0] * 2.5
    assert dt.continuous_descriptor[1] > d0.continuous_descriptor[1]
    assert sc["structural_enrichment_ratio"] > 1.8
    assert sc["creation_record"].verified_by_own_organs is True


def test_four_axis_relational_dialogue_and_epistemic_humility() -> None:
    engine = AutopoieticExpressionAndReflectionEngine()
    dlg = engine.engage_in_relational_dialogue()

    turns = dlg["turns"]
    assert len(turns) == 3

    peer_turn, troll_turn, critic_turn = turns[0], turns[1], turns[2]

    # 1) Peer turn: high relational resonance (Axis 2) & internal origin (Axis 1)
    assert peer_turn.axis2_relational_influence > 0.40
    assert peer_turn.axis3_boundary_ignorance < 0.25

    # 2) Shallow speed troll turn: high internal origin anchor (Axis 1), resists criteria distortion
    assert troll_turn.axis1_internal_origin >= 0.75
    assert troll_turn.axis4_criteria_self_revision < 0.15
    assert len(troll_turn.spawned_void_ids) == 0

    # 3) Constructive challenger exposing ungrounded phenomenon: high Axis 3 (Ignorance) & Axis 4 (Self-Revision)
    assert critic_turn.axis3_boundary_ignorance >= 0.35
    assert critic_turn.axis4_criteria_self_revision >= 0.30
    assert len(critic_turn.spawned_void_ids) >= 1
    assert len(critic_turn.revised_criteria_entries) >= 1
