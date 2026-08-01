import pytest
import os
import numpy as np
from core.evolution.conceptual_causal_gear import ConceptualCausalGear
from core.evolution.moulting_plasticity import MoultingPlasticityEngine
from core.memory.causal_controller import CausalMemoryController


def test_conceptual_causal_gear_stereoscopic_alignment():
    """
    Verifies that ConceptualCausalGear correctly implements:
    1. Stereoscopic Triangulation (using left focus memory prior and right focus predicted outcome).
    2. Disparity angle calculation.
    3. Causal Depth triangulation.
    4. Active Partitioning (Connection vs Separation).
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)
    plasticity = MoultingPlasticityEngine(mc, dimensions=3)
    gear = ConceptualCausalGear(mc, plasticity)

    # 1. Let's align 'bird' concept
    res_bird = gear.process_and_align_concept(
        concept_key="bird",
        world_description="A beautiful alive bird is spreading its wings and soaring high into the sky",
        raw_stimulus=b"\x0a\x0b\x0c"
    )

    # Asserts on stereoscopic triangulation parameters
    assert "anchor_left" in res_bird
    assert "anchor_right" in res_bird
    assert "world_vector" in res_bird
    assert "causal_depth" in res_bird
    assert "disparity_angle" in res_bird
    assert "connection_ratio" in res_bird
    assert "separation_tension" in res_bird

    # Bird flew beautifully, so connection ratio should be quite high
    assert res_bird["connection_ratio"] > 0.0

    # 2. Let's process a stone-bird (heavy, gravity, stone) to trigger high separation tension
    res_heavy = gear.process_and_align_concept(
        concept_key="bird",
        world_description="This bird is heavy as a cold stone, locked by strong gravity to the ground, dead",
        raw_stimulus=b"\xff\x00\xff"
    )

    # The heavy bird contradicts the fly/wing prior, so separation tension should be significant
    assert res_heavy["separation_tension"] > 0.1
    # Check that triangulation still computed causal depth
    assert res_heavy["causal_depth"] > 0.0


def test_conceptual_causal_gear_unseen_concept():
    """
    Verifies that ConceptualCausalGear dynamically seeds unseen words
    as new causes and aligns them stereoscopically without errors.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)
    gear = ConceptualCausalGear(mc)

    concept = "quantum_falcon"
    assert concept not in gear.internal_cause_registry

    # Process
    res = gear.process_and_align_concept(
        concept_key=concept,
        world_description="A swift quantum falcon slicing coordinates with precise movement",
        raw_stimulus=b"spark"
    )

    # New concept should now be registered and tuned
    assert concept in gear.internal_cause_registry
    assert len(gear.internal_cause_registry[concept]) == 4
