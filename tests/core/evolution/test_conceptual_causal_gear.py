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
    assert "attention_lens_vector" in res_bird

    # Bird flew beautifully, so connection ratio should be quite high
    assert res_bird["connection_ratio"] > 0.0

    # 2. Let's process a stone-bird (heavy, gravity, stone) to trigger high separation tension
    res_heavy = gear.process_and_align_concept(
        concept_key="bird",
        world_description="This bird is heavy as a cold stone, locked by strong gravity to the ground, dead",
        raw_stimulus=b"\xff\x00\xff"
    )

    # The heavy bird contradicts the fly/wing prior, so separation tension should be significant
    assert res_heavy["separation_tension"] > 0.0
    # Check that triangulation still computed causal depth
    assert res_heavy["causal_depth"] > 0.0


def test_attention_structure_as_connection_criteria():
    """
    Proves that the Informational Attention Lens (선택과 집중) acts as the criteria
    of connection, changing which dimensions get connected or separated.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)
    gear = ConceptualCausalGear(mc)

    # Dynamic Focus: 1. Focus on "Rise" (wings/sky description)
    res_rise = gear.process_and_align_concept(
        concept_key="bird",
        world_description="The wings flap, soaring into the sky, rising high",
        raw_stimulus=b"\x01"
    )
    # The lens should shift to prioritize "Rise" (index 1 is 0.70)
    assert gear.attention_lens_vector[1] == 0.70

    # Dynamic Focus: 2. Focus on "Life" (alive/creature description)
    res_life = gear.process_and_align_concept(
        concept_key="bird",
        world_description="A beautiful alive and breathing creature with deep life energy",
        raw_stimulus=b"\x02"
    )
    # The lens should shift to prioritize "Life" (index 3 is 0.70)
    assert gear.attention_lens_vector[3] == 0.70
