"""
Verification suite for ELYSIA StructuralScaleRotorCortex:
Non-numeric, causal-structural Scale-Hierarchy Rotor where Physics, Chemistry,
DNA/Biology, Mathematics, Code Architecture, Language, and Real Workspace Code
observe, classify, re-recognize, revise criteria, and recombine with one another.
"""

from __future__ import annotations

from pathlib import Path

from core.topology.structural_scale_rotor import (
    CausalRelation,
    CausalStructure,
    CausalVerb,
    build_cross_domain_reference_atlas,
    extract_structure_from_prose,
    extract_structure_from_python_file,
)


def test_everything_observes_and_discerns_everything_across_physics_chemistry_dna_code_language():
    """DNA proofreading lens observes Code & Chemistry purely by shared causal topology."""
    cortex = build_cross_domain_reference_atlas()

    rec_rollback = cortex.observe_through("code::transactional_rollback", "dna::proofreading")
    rec_buffer = cortex.observe_through("chemistry::buffer_equilibrium", "dna::proofreading")
    rec_fragile = cortex.observe_through("code::fragile_linear_pipeline", "dna::proofreading")

    assert rec_rollback.structural_resonance > 0.75
    assert rec_buffer.structural_resonance > 0.70
    assert rec_fragile.structural_resonance < 0.35
    assert rec_rollback.structural_resonance > rec_fragile.structural_resonance + 0.40

    # Explicit structural role correspondence mapped onto DNA proofreading error correction
    assert any(m.shared_verb == "feeds_back_negative" for m in rec_rollback.role_mappings)
    assert "feeds_back_negative" in rec_fragile.target_deficits_vs_lens


def test_perspective_rotation_re_recognizes_same_phenomenon_as_different_causal_definitions():
    """Rotating cross-domain lenses over one target yields distinct definitions and blind spots."""
    cortex = build_cross_domain_reference_atlas()

    rot = cortex.rotate_perspectives(
        target_id="chemistry::water_hydrogen_network",
        lens_ids=[
            "physics::harmonic_oscillator",
            "dna::proofreading",
            "language::metaphor_bridge",
            "code::fragile_linear_pipeline",
        ],
    )

    recs = rot["recognitions"]
    assert len(recs) == 4
    # Metaphor bridge catches `bridges_across`, whereas fragile_linear_pipeline is blind to binding/bridges/opposes
    frag_rec = recs[3]
    assert "bridges_across" in frag_rec.lens_blind_spots
    assert "binds_with" in frag_rec.lens_blind_spots
    assert "chemistry::water_hydrogen_network" in rot["integrated_definition_ko"]
    assert len(set(rot["distinct_shared_verb_sets"])) >= 3


def test_cross_domain_recombination_heals_fragile_code_with_dna_or_chemistry_principle():
    """Diagnoses brittle one-way code and transplants cross-domain negative feedback to stabilize it."""
    cortex = build_cross_domain_reference_atlas()

    res = cortex.diagnose_and_recombine("code::fragile_linear_pipeline", cross_domain_only=True)
    assert res.donor_domain != "code"
    assert res.resilience_before is False
    assert res.resilience_after is True
    assert res.created_structure.has_stabilizing_feedback() is True
    assert res.created_structure.is_runaway_fragile() is False
    assert res.created_structure.structure_id in cortex.structures


def test_cross_domain_recombination_heals_escalating_language_conflict_with_cross_domain_principle():
    """Heals runaway positive-feedback language conflict using physics/chemistry/DNA balance."""
    cortex = build_cross_domain_reference_atlas()

    res = cortex.diagnose_and_recombine("language::escalating_conflict", cross_domain_only=True)
    assert res.donor_domain != "language"
    assert res.resilience_before is False
    assert res.resilience_after is True
    assert res.created_structure.has_stabilizing_feedback() is True
    assert res.created_structure.is_runaway_fragile() is False


def test_criterion_self_revision_when_lens_fails_to_distinguish_two_structures():
    """A coarse 1-verb criterion expands itself when blind to the difference between resilient and fragile code."""
    cortex = build_cross_domain_reference_atlas()
    cortex.register_structure(
        CausalStructure(
            structure_id="criterion::coarse_throughput_only",
            domain="meta_criterion",
            scale_tier="meso",
            micro_constituents=("throughput_step",),
            process_chain=(CausalRelation("input", CausalVerb.DRIVES, "output", "meso"),),
            macro_phenomenon="단순 처리 속도만 재는 조악한 1차원 기준",
            observing_perspective="naive_throughput_only",
            observing_criterion="drives_only",
        )
    )

    rev = cortex.revise_criterion_from_friction(
        lens_id="criterion::coarse_throughput_only",
        target_a_id="code::transactional_rollback",
        target_b_id="code::fragile_linear_pipeline",
    )

    assert rev.resonance_gap_before < 0.15
    assert rev.resonance_gap_after > rev.resonance_gap_before + 0.25
    assert "feeds_back_negative" in rev.added_verbs
    assert "revises_criterion" in rev.added_verbs


def test_real_workspace_python_ast_and_korean_prose_enter_structural_rotor():
    """Real workspace Python AST and Korean causal prose enter the same non-numeric rotor."""
    cortex = build_cross_domain_reference_atlas()

    real_py = extract_structure_from_python_file(
        Path(r"c:\Elysia\L1_causal_field\core\topology\causal_scale_rotor.py")
    )
    korean_prose = extract_structure_from_prose(
        structure_id="prose::cell_homeostasis",
        domain="biology_prose",
        prose=(
            "외부 자극이 세포막 수용체를 구동한다. "
            "세포 경계막이 유해 물질 유입을 억제하고 내부 효소가 손상을 교정한다. "
            "신호 전달이 핵심 경로를 연결해 생명 평형을 이룬다."
        ),
    )

    cortex.register_structure(real_py)
    cortex.register_structure(korean_prose)

    rec_py_dna = cortex.observe_through(real_py.structure_id, "dna::proofreading")
    rec_py_phys = cortex.observe_through(real_py.structure_id, "physics::harmonic_oscillator")
    rec_prose_dna = cortex.observe_through(korean_prose.structure_id, "dna::proofreading")

    assert len(real_py.micro_constituents) >= 1
    assert rec_py_dna.structural_resonance > 0.35
    assert rec_py_phys.structural_resonance > 0.25
    assert rec_prose_dna.structural_resonance > 0.45


def test_classify_entire_universe_by_any_chosen_principle_as_axis():
    """Classifies all registered structures across all domains using DNA proofreading or Chemistry catalysis."""
    cortex = build_cross_domain_reference_atlas()

    ranked_by_dna = cortex.classify_all_by_lens("dna::proofreading")
    ranked_by_cat = cortex.classify_all_by_lens("chemistry::catalysis")

    assert len(ranked_by_dna) == len(cortex.structures) - 1
    # Strictly sorted descending by resonance
    scores_dna = [r.structural_resonance for r in ranked_by_dna]
    assert scores_dna == sorted(scores_dna, reverse=True)
    # Under Catalysis (`bridges_across` + `binds_with`), metaphor_bridge & water_hydrogen_network outrank fragile_linear_pipeline
    score_map_cat = {r.target_id: r.structural_resonance for r in ranked_by_cat}
    assert score_map_cat["language::metaphor_bridge"] > score_map_cat["code::fragile_linear_pipeline"]
    assert score_map_cat["chemistry::water_hydrogen_network"] > score_map_cat["code::fragile_linear_pipeline"]
