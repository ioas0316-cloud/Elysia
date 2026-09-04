"""
Unit and Integration Tests for Self-Referential Information Architecture Engine
================================================================================
"""

import numpy as np
import pytest

from core.topology.self_referential_architecture import (
    CausalEngine0,
    CausalDeformationLayer,
    PrimitiveDiscernmentEngine,
    SelfReferentialLanguageEngine,
    SelfReferentialVideoEngine,
    MetaOperator,
    MetaDefinition,
    RecursiveCausalityLoop,
    VolitionalGeometricRotorEngine,
    MetaCognitiveDomainLensSwitcher,
    FoundationalArchetypeDecodingEngine,
    DimensionalCircuit,
    MetaResonanceBus,
    DynamicDimensionSelfDifferentiationEngine,
    LabelSelfAssimilationEngine,
    SelfReferentialArchitectureEngine,
)


def test_causal_engine_0_convergence():
    engine = CausalEngine0(dim=3)
    intent = np.array([2.0, -1.0, 3.0])

    initial_state, initial_delta = engine.cycle(intent, lr=0.1)

    for _ in range(50):
        current_state, delta = engine.cycle(intent, lr=0.1)

    assert delta < initial_delta
    assert delta < 0.5


def test_causal_deformation_layer_relaxation_and_multilayer():
    layer1 = CausalDeformationLayer(in_dim=4, out_dim=3)
    layer2 = CausalDeformationLayer(in_dim=3, out_dim=3)
    input_intent = np.array([1.5, -0.5, 2.0, 0.1])

    s1, r1 = layer1.relax_and_update(input_intent, relaxation_steps=5)
    s2, r2 = layer2.relax_and_update(s1, relaxation_steps=5)

    # Bi-directional standing wave feedback
    s1_res, r1_res = layer1.relax_and_update(
        input_intent, higher_friction_R=np.array([r2, r2, r2]), relaxation_steps=3
    )

    assert len(s1_res) == 3
    assert r1_res >= 0.0


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

    normal_res = engine.verify_and_reject_anomaly({"digit_count": 5, "kinematic_stress": 0.1})
    assert normal_res["is_rejected"] is False

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


def test_dimensional_circuits_and_meta_resonance_bus():
    lang_c = DimensionalCircuit("Language", lambda s: float(np.std(s) * 0.5))
    math_c = DimensionalCircuit("Math", lambda s: float(abs(np.sum(s) - 1.0)))
    bus = MetaResonanceBus()

    s_lang, r_lang = lang_c.step(np.array([1.0, 0.5, 0.0, 0.0]))
    assert r_lang >= 0.0

    s_math, r_math = bus.phase_shift_coupling(s_lang, r_lang, math_c)
    assert r_math >= 0.0

    total_r = bus.compute_total_resonance_friction({"Lang": r_lang, "Math": r_math})
    assert total_r == pytest.approx(r_lang + r_math)


def test_dynamic_dimension_self_differentiation():
    sprouter = DynamicDimensionSelfDifferentiationEngine(unmapped_threshold=0.5)

    no_sprout = sprouter.evaluate_and_sprout(0.3, "SubtleNoise")
    assert no_sprout is None

    sprouted = sprouter.evaluate_and_sprout(0.8, "Emotional_Resonance")
    assert sprouted is not None
    assert "SproutedDim_Emotional_Resonance_1" in sprouted.name


def test_label_self_assimilation_engine():
    engine = LabelSelfAssimilationEngine()
    res = engine.reverse_engineer_label(
        external_label="로그 (Logarithm)",
        observed_phenomenon={
            "constraints": np.array([2.0, 1.0]),
            "effect_trajectory": np.array([0.5, 0.2])
        }
    )
    assert res["external_label"] == "로그 (Logarithm)"
    assert res["is_assimilated_as_internal_knowledge"] is True
    assert "결과의 증상" in res["self_assimilation_proof"]


def test_full_self_referential_architecture_cycle():
    full_engine = SelfReferentialArchitectureEngine()
    output = full_engine.run_full_self_referential_cycle({
        "video_data": {"digit_count": 6, "kinematic_stress": 0.8},
        "unmapped_friction": 0.9,
        "external_label": "Gravity",
        "voltage_intent": np.array([2.0, -1.0, 3.0]),
        "layer1_intent": np.array([1.5, -0.5, 2.0, 0.1])
    })

    assert "causal_engine_0_equilibrium" in output
    assert "multi_layer_resonance_friction" in output
    assert output["0th_primitive_discernment"]["shared_coherence"] is True
    assert output["video_self_rejection"]["is_rejected"] is True
    assert output["label_self_assimilation"]["is_assimilated_as_internal_knowledge"] is True
    assert "Emotional_Resonance" in output["sprouted_dimension"]
