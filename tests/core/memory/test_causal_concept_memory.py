"""
Tests for causal concept memory and variant discernment.
"""

from core.memory.causal_concept_memory import (
    CausalConceptMemory,
    CausalObservation,
    VariantReality,
)


def _apple_memory():
    memory = CausalConceptMemory(vector_dim=8)
    invariants = [
        "botanical_fruit_body",
        "seed_bearing_organism",
        "sugar_acid_aroma_chemistry",
        "edible_biological_context",
    ]
    memory.learn_observation(
        CausalObservation(
            observation_id="real_red_apple",
            concept_id="apple",
            features={
                "kind": "fruit",
                "color": "red",
                "texture": "crisp",
                "chemistry": "sugar_acid_aroma",
            },
            process_context={"domain_layer": "chemistry_layer", "growth": "orchard_tree"},
            evidence_strength=1.0,
            is_real_world=True,
        ),
        label="Apple",
        invariants=invariants,
    )
    memory.learn_observation(
        CausalObservation(
            observation_id="real_green_apple",
            concept_id="apple",
            features={
                "kind": "fruit",
                "color": "green",
                "texture": "crisp",
                "chemistry": "sugar_acid_aroma",
            },
            process_context={"domain_layer": "chemistry_layer", "growth": "orchard_tree"},
            evidence_strength=0.9,
            is_real_world=True,
        ),
        label="Apple",
        invariants=invariants,
    )
    return memory


def test_apple_definition_becomes_causal_memory_not_label_only():
    memory = _apple_memory()
    concept = memory.concepts["apple"]

    assert concept.label == "Apple"
    assert "seed_bearing_organism" in concept.invariants
    assert "color:red" in concept.feature_attractors
    assert "color:green" in concept.feature_attractors
    assert concept.feature_attractors["chemistry:sugar_acid_aroma"].support_mass > 0.0
    assert len(concept.causal_history) == 2


def test_real_red_apple_is_grounded_by_observed_causal_attractors():
    memory = _apple_memory()

    assessment = memory.assess_variant(
        "apple",
        "red_apple",
        {
            "kind": "fruit",
            "color": "red",
            "texture": "crisp",
            "chemistry": "sugar_acid_aroma",
        },
        {"variable_axes": ["color"], "domain_layer": "chemistry_layer"},
    )

    assert assessment.reality is VariantReality.REAL_GROUNDED
    assert assessment.grounded_support > 0.0
    assert assessment.evidence["structural_validity"] > 0.0


def test_blue_apple_can_be_imaginable_without_reality_grounding():
    memory = _apple_memory()

    assessment = memory.assess_variant(
        "apple",
        "blue_apple",
        {
            "kind": "fruit",
            "color": "blue",
            "texture": "crisp",
            "chemistry": "sugar_acid_aroma",
        },
        {"variable_axes": ["color"], "domain_layer": "chemistry_layer"},
    )

    assert assessment.reality in {VariantReality.REAL_GROUNDED, VariantReality.IMAGINABLE}
    assert assessment.imagination_support > 0.0
    assert assessment.contradiction_tension < 1.0


def test_variant_that_breaks_apple_invariants_is_contradicted():
    memory = _apple_memory()

    assessment = memory.assess_variant(
        "apple",
        "metal_apple",
        {
            "kind": "machine_part",
            "color": "silver",
            "texture": "machined",
            "chemistry": "iron_alloy",
        },
        {
            "variable_axes": ["color"],
            "domain_layer": "physics_layer",
            "violates_invariants": ["seed_bearing_organism", "sugar_acid_aroma_chemistry"],
        },
    )

    assert assessment.reality is VariantReality.CONTRADICTED
    assert assessment.contradiction_tension >= 1.0
