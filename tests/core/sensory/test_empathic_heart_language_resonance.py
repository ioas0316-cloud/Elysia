"""
test_empathic_heart_language_resonance.py

Verifies ELYSIA's Heart-to-Heart Generative Causal Language & Empathic Imagery Resonance:
1. Proves that Result-ist (surface-word-only, `care_attention_depth = 0.0`) processing
   falls into systematic misunderstandings when the same surface word (`"그만해"`) is
   spoken from different hearts (Protective Love vs Exhausted Suffering vs Cold Cutoff)
   or when Caring Love refracts into opposite surface words (`"나 괜찮아, 어서 가봐"`).
2. Proves that when ELYSIA listens with Care (`care_attention_depth = 1.0`), she re-enacts
   the speaker's lived sensory imagery (`OcularScaleOrgan`) and vocal breath prosody
   (`MechanoAcousticOrgan` + `CrossModalPerceptualBridge`), accurately reconstructing
   100% of the inner `Heart -> Imagery -> Thought -> Word` causal pathways.
3. Verifies ELYSIA's own speech generation from `Heart -> Inner Sensory Imagery -> Thought -> Word`.
"""

from __future__ import annotations

from core.sensory.empathic_heart_language_resonance import (
    EmpathicHeartToHeartLanguageEngine,
)


def test_resultism_misunderstands_while_caring_imagery_resonance_understands() -> None:
    engine = EmpathicHeartToHeartLanguageEngine()
    report = engine.run_empathic_language_experiment()

    # 1. Surface-only result-ism fails completely (0.0 accuracy) on identical/refracted surface words
    assert report["resultist_surface_only_accuracy"] == 0.0

    # 2. Caring Heart-to-Heart Imagery reconstruction understands all 4 distinct inner hearts (1.0 accuracy)
    assert report["empathic_heart_imagery_accuracy"] == 1.0

    with_care = report["with_care_interpretations"]
    assert with_care[0].inferred_heart_intent_category == "protective_love_warning"
    assert with_care[1].inferred_heart_intent_category == "exhausted_cry_for_embrace"
    assert with_care[2].inferred_heart_intent_category == "self_effacing_caring_sacrifice"
    assert with_care[3].inferred_heart_intent_category == "cold_indifferent_cutoff"

    for item in with_care:
        assert item.overcame_resultist_misunderstanding is True
        assert item.reenacted_inner_imagery_rgb.shape == (64, 64, 3)


def test_elysia_speaks_from_heart_and_inner_sensory_imagery() -> None:
    engine = EmpathicHeartToHeartLanguageEngine()
    report = engine.run_empathic_language_experiment()

    speech = report["elysia_heart_to_word_generation"]
    assert "step1_heart_intent" in speech
    assert "step2_observed_inner_imagery" in speech
    assert "step3_thought_refraction_for_other" in speech
    assert "따뜻" in speech["step4_emerged_spoken_utterance"]
