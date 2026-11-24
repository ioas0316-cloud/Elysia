"""
Verification suite for ELYSIA Phase 12:
Living Causal Text Law Synthesizer, Dimensional Mirror, Multi-World Constant
Self-Discovery & Metaphor Self-Reflection.
"""

from __future__ import annotations

import math

from core.orchestration.causal_text_law_synthesizer import (
    CausalTextLawSynthesizer,
    TextSynthesisOutcome,
    sync_synthesis_to_causal_field,
)
from core.orchestration.extended_cognitive_organism import (
    ExtendedCognitiveOrganism,
    ExtendedWorldState,
    WIEN_CONSTANT_NM_K,
)
from core.physics.causal_field import CausalField


def _wien_calibration_worlds():
    return [
        ExtendedWorldState(source_temp_k=3200.0),
        ExtendedWorldState(source_temp_k=5800.0),
    ]


def test_explicit_equation_text_parsed_verified_and_internalized():
    synth = CausalTextLawSynthesizer()
    org = ExtendedCognitiveOrganism()
    org.graft_spectrometer()

    text = "빈의 변위 법칙에 따르면 방출 파장과 온도의 곱은 일정하다: `lambda * T = 2897771.955`"
    res = synth.synthesize_and_verify(
        text=text,
        organism=org,
        calibration_worlds=_wien_calibration_worlds(),
    )
    assert res.outcome == TextSynthesisOutcome.VERIFIED_PHYSICAL_LAW
    assert res.multi_world_action < 1e-8
    assert res.internalized_law_name in org.internalized_disciplines

    # Detach spectrometer AND blind eye on unseen 4500K world -> zero-shot prediction
    org.detach_tool("spectrometer")
    unseen = ExtendedWorldState(source_temp_k=4500.0)
    perc = org.perceive_extended(unseen, active_organs=("ear", "hand"))
    assert math.isclose(
        perc.inferred_world["emission_wavelength_nm"],
        WIEN_CONSTANT_NM_K / 4500.0,
        abs_tol=1e-2,
    )


def test_equation_free_natural_language_discovers_wien_constant_autonomously():
    synth = CausalTextLawSynthesizer()
    org = ExtendedCognitiveOrganism()
    org.graft_spectrometer()

    # Pure Korean prose with ZERO numbers or formulas
    prose = "물체의 절대온도가 높아질수록 최대 방출 파장은 온도에 반비례한다."
    res = synth.synthesize_and_verify(
        text=prose,
        organism=org,
        calibration_worlds=_wien_calibration_worlds(),
    )
    assert res.outcome == TextSynthesisOutcome.VERIFIED_PHYSICAL_LAW
    assert res.multi_world_action < 1e-8
    assert res.discovered_constant is not None
    rel_err = abs(res.discovered_constant - WIEN_CONSTANT_NM_K) / WIEN_CONSTANT_NM_K
    assert rel_err < 1e-4

    # Detach tool and predict unseen 4500K world purely from internalized prose law
    org.detach_tool("spectrometer")
    unseen = ExtendedWorldState(source_temp_k=4500.0)
    perc = org.perceive_extended(unseen, active_organs=("ear", "hand"))
    assert math.isclose(
        perc.inferred_world["emission_wavelength_nm"],
        WIEN_CONSTANT_NM_K / 4500.0,
        abs_tol=1e-1,
    )


def test_multi_world_reproducibility_falsifies_proportional_myth():
    synth = CausalTextLawSynthesizer()
    org = ExtendedCognitiveOrganism()
    org.graft_spectrometer()

    myth = "물체의 온도가 높아질수록 최대 방출 파장도 온도에 비례해서 커진다."
    res = synth.synthesize_and_verify(
        text=myth,
        organism=org,
        calibration_worlds=_wien_calibration_worlds(),
    )
    assert res.outcome == TextSynthesisOutcome.EMPIRICALLY_FALSIFIED
    assert res.experiment_skipped is False
    assert res.multi_world_action > 0.05
    assert len(org.internalized_disciplines) == 0


def test_dimensional_mirror_rejects_incompatible_units_before_experiment():
    synth = CausalTextLawSynthesizer()
    org = ExtendedCognitiveOrganism()
    org.graft_spectrometer()

    for bad_eq in ("파장 + 온도 = 500", "lambda + T = 100"):
        res = synth.synthesize_and_verify(
            text=bad_eq,
            organism=org,
            calibration_worlds=_wien_calibration_worlds(),
        )
        assert res.outcome == TextSynthesisOutcome.DIMENSIONAL_CONTRADICTION
        assert res.experiment_skipped is True
        assert "[L]" in res.claim.dimension_error
        assert "[Theta]" in res.claim.dimension_error


def test_inverse_square_sound_sentence_verified_across_distances():
    synth = CausalTextLawSynthesizer()
    org = ExtendedCognitiveOrganism()

    worlds = [
        ExtendedWorldState(source_pos_x=5.0, source_power_w=40.0),
        ExtendedWorldState(source_pos_x=10.0, source_power_w=40.0),
    ]
    prose = "거리가 멀어질수록 소리 세기는 거리의 제곱에 반비례한다."
    res = synth.synthesize_and_verify(
        text=prose,
        organism=org,
        calibration_worlds=worlds,
        use_spectrometer=False,
    )
    assert res.outcome == TextSynthesisOutcome.VERIFIED_PHYSICAL_LAW
    assert res.multi_world_action < 1e-8
    assert res.discovered_constant is not None
    expected_c = 40.0 / (4.0 * math.pi)
    assert math.isclose(res.discovered_constant, expected_c, abs_tol=1e-3)


def test_poetic_metaphor_honest_literal_limit_and_physical_backtracing():
    synth = CausalTextLawSynthesizer()
    org = ExtendedCognitiveOrganism()

    res_fire = synth.synthesize_and_verify(
        text="그의 분노는 붉은 용광로처럼 뜨겁게 타오른다.",
        organism=org,
        calibration_worlds=_wien_calibration_worlds(),
        use_spectrometer=False,
    )
    assert res_fire.outcome == TextSynthesisOutcome.ABSTRACT_METAPHOR_REFLECTION
    assert res_fire.experiment_skipped is True
    assert res_fire.metaphor_reflection is not None
    assert "분노" in res_fire.metaphor_reflection.abstract_targets
    assert res_fire.metaphor_reflection.grounded_domain == "thermal_radiant_source"
    assert res_fire.metaphor_reflection.chromatic_transfer["red_flux"] > 0.6

    res_stone = synth.synthesize_and_verify(
        text="침묵은 무거운 돌처럼 방 안에 내려앉았다.",
        organism=org,
        calibration_worlds=_wien_calibration_worlds(),
        use_spectrometer=False,
    )
    assert res_stone.outcome == TextSynthesisOutcome.ABSTRACT_METAPHOR_REFLECTION
    assert res_stone.metaphor_reflection is not None
    assert "침묵" in res_stone.metaphor_reflection.abstract_targets
    assert res_stone.metaphor_reflection.grounded_domain == "gravitational_inertia"
    assert res_stone.metaphor_reflection.chromatic_transfer["blue_order"] > 0.6


def test_mixed_document_ingestion_separates_all_four_outcomes():
    synth = CausalTextLawSynthesizer()
    org = ExtendedCognitiveOrganism()
    org.graft_spectrometer()

    doc = "\n".join(
        [
            "물체의 절대온도가 높아질수록 최대 방출 파장은 온도에 반비례한다.",
            "파장 + 온도 = 500",
            "물체의 온도가 높아질수록 최대 방출 파장도 온도에 비례해서 커진다.",
            "그의 분노는 붉은 용광로처럼 뜨겁게 타오른다.",
        ]
    )
    report = synth.ingest_document_text(
        document_text=doc,
        organism=org,
        calibration_worlds=_wien_calibration_worlds(),
    )
    assert len(report.verified_laws) == 1
    assert len(report.dimensional_rejections) == 1
    assert len(report.falsified_claims) == 1
    assert len(report.metaphor_reflections) == 1
    assert len(org.internalized_disciplines) == 1


def test_safe_ast_rejects_code_injection_and_syncs_causal_field():
    synth = CausalTextLawSynthesizer()
    org = ExtendedCognitiveOrganism()

    inj = synth.synthesize_and_verify(
        text="__import__('os').system('echo pwn') = 1",
        organism=org,
        calibration_worlds=_wien_calibration_worlds(),
        use_spectrometer=False,
    )
    assert inj.outcome == TextSynthesisOutcome.UNPARSEABLE_OR_UNSAFE
    assert inj.experiment_skipped is True

    org.graft_spectrometer()
    doc = "\n".join(
        [
            "물체의 절대온도가 높아질수록 최대 방출 파장은 온도에 반비례한다.",
            "그의 분노는 붉은 용광로처럼 뜨겁게 타오른다.",
        ]
    )
    report = synth.ingest_document_text(doc, org, _wien_calibration_worlds())
    sync_res = sync_synthesis_to_causal_field(report, causal_field=CausalField(dimensions=3), prefix="txt")
    assert sync_res["abstract_sky_curiosity"] == 1.0
    sync_res["causal_field"].step(dt=0.1)
