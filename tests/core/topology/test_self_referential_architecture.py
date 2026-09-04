"""
Unit and Integration Tests for Self-Referential Information Architecture Engine
================================================================================
"""

import numpy as np
import pytest

from core.topology.self_referential_architecture import (
    PrimitiveDiscernmentEngine,
    SelfReferentialLanguageEngine,
    SelfReferentialVideoEngine,
    MetaOperator,
    MetaDefinition,
    RecursiveCausalityLoop,
    VolitionalGeometricRotorEngine,
    MetaCognitiveDomainLensSwitcher,
    FoundationalArchetypeDecodingEngine,
    KnowledgeDimensionPhaseTransitionEngine,
    SelfReferentialArchitectureEngine,
)


def test_primitive_discernment_1_vs_2():
    engine = PrimitiveDiscernmentEngine()
    u1 = engine.create_unity("1", dim=4)
    assert u1.dimension_rank == 1
    assert u1.internal_prototype is None

    u2 = engine.expand_structure(u1, "+1")
    assert u2.dimension_rank == 2
    assert u2.internal_prototype == u1

    result = engine.discern_difference(u1, u2)
    assert result["shared_coherence"] is True
    assert result["rank_difference"] == 1
    assert result["phase_discrepancy"] > 0.0


def test_self_referential_language():
    engine = SelfReferentialLanguageEngine()
    res = engine.redefine_term_in_context("자아", ["의지", "인과"], ["시스템", "연산"])
    assert "term" in res
    assert res["term"] == "자아"
    assert "semantic_friction" in res
    assert "redefined_value" in res


def test_self_referential_video_anomaly_rejection():
    engine = SelfReferentialVideoEngine(max_fingers_allowed=5)

    # 5 fingers -> normal
    normal_res = engine.verify_and_reject_anomaly({"digit_count": 5, "kinematic_stress": 0.1})
    assert normal_res["is_rejected"] is False

    # 6 fingers -> rejected due to structural friction
    anomalous_res = engine.verify_and_reject_anomaly({"digit_count": 6, "kinematic_stress": 0.5})
    assert anomalous_res["is_rejected"] is True
    assert "6번째 손가락" in anomalous_res["rejection_reason"]


def test_meta_operator_and_meta_definition():
    op1 = MetaOperator("+", binding_power=1.0)
    op2 = MetaOperator("*", binding_power=2.0)
    meta_op = op1.apply_meta_transformation(op2, causal_constraint=0.5)

    assert meta_op.symbol == "[+⊗*]"
    assert meta_op.binding_power == (1.0 + 2.0) * 1.5

    meta_def = MetaDefinition(
        intention=np.array([1.0, 0.8, -0.5]),
        constraints=np.array([0.2, 0.4, 0.1])
    )
    def_res = meta_def.execute_causal_process()
    assert "output_result" in def_res
    assert "process_friction" in def_res


def test_recursive_causality_loop():
    initial_boundary = np.array([1.0, 1.0, 1.0, 1.0])
    loop = RecursiveCausalityLoop(initial_boundary)

    raw_cause = np.array([0.8, 0.3, 0.9, 0.2])
    internal_ground = np.array([0.1, 0.1, 0.1, 0.1])

    cycle1 = loop.execute_cycle(raw_cause, internal_ground)
    bnd_after_c1 = cycle1["updated_boundary_constraint"]

    # Subsequent cycle uses the updated boundary
    cycle2 = loop.execute_cycle(raw_cause, internal_ground)
    bnd_after_c2 = cycle2["updated_boundary_constraint"]

    assert not np.array_equal(bnd_after_c1, bnd_after_c2)


def test_volitional_geometric_rotor():
    rotor = VolitionalGeometricRotorEngine()
    intention = np.array([1.0, 1.0, 0.0, 0.0])
    current_state = np.array([0.2, 0.5, 0.0, 0.0])

    volition = rotor.compute_volition_vector(intention, current_state)
    assert np.array_equal(volition, np.array([0.8, 0.5, 0.0, 0.0]))

    cand_a = np.array([0.8, 0.5, 0.0, 0.0])
    cand_b = np.array([0.0, 0.0, 1.0, 0.0])
    constraints = np.array([0.1, 0.1, 0.1, 0.0])

    res = rotor.compare_rotor_trajectories(volition, cand_a, cand_b, constraints)
    assert res["chosen_trajectory"] == "Candidate A"
    assert "hypothetical_rotation_axis" in res


def test_meta_cognitive_lens_switcher_and_cross_dimensional():
    switcher = MetaCognitiveDomainLensSwitcher()
    assert switcher.detect_and_switch_lens("Linguistic word context") == "Linguistic Lens"
    assert switcher.detect_and_switch_lens("Geometry vector math") == "Geometric/Mathematical Lens"
    assert switcher.detect_and_switch_lens("3D video physics") == "Physical/Sensory Lens"

    proj = switcher.project_cross_dimensions("빛 (Light)")
    assert len(proj["projections"]) == 5
    assert "Particle" in proj["projections"]
    assert "Wave" in proj["projections"]
    assert "Language" in proj["projections"]


def test_foundational_archetype_decoding():
    decoder = FoundationalArchetypeDecodingEngine()
    res = decoder.translate_unknown_domain("세포생물학", "수송체 막 전이")
    assert "structural_translation" in res
    assert "combinatorial_understanding" in res


def test_knowledge_dimension_phase_transition():
    engine = KnowledgeDimensionPhaseTransitionEngine()
    res = engine.execute_knowledge_phase_shift("의지적 자 자율 인지")
    assert len(res["phase_shift_sequence"]) == 3
    assert "LanguageDim" in res["phase_shift_sequence"][0]
    assert "MathDim" in res["phase_shift_sequence"][1]
    assert "PhysicsDim" in res["phase_shift_sequence"][2]


def test_full_self_referential_architecture_cycle():
    full_engine = SelfReferentialArchitectureEngine()
    output = full_engine.run_full_self_referential_cycle({
        "video_data": {"digit_count": 6, "kinematic_stress": 0.8}
    })

    assert output["0th_primitive_discernment"]["shared_coherence"] is True
    assert output["video_self_rejection"]["is_rejected"] is True
    assert output["volitional_rotor_exploration"]["chosen_trajectory"] == "Candidate A"
    assert len(output["cross_dimensional_projection"]["projections"]) == 5
