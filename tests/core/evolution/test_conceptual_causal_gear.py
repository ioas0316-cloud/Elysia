import pytest
import os
import numpy as np
from core.evolution.conceptual_causal_gear import ConceptualCausalGear
from core.evolution.moulting_plasticity import MoultingPlasticityEngine
from core.memory.causal_controller import CausalMemoryController


def test_conceptual_causal_gear_alignment():
    """
    Verifies that ConceptualCausalGear correctly maps a concept (e.g. 'bird'),
    computes internal cause, predicted outcome, and compared world fact,
    and dynamically adjusts the internal cause register (cognitive tuning).
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)
    plasticity = MoultingPlasticityEngine(mc, dimensions=3)
    gear = ConceptualCausalGear(mc, plasticity)

    # 1. Inspect original 'bird' representation
    original_bird = gear.internal_cause_registry["bird"].copy()
    assert original_bird[0] == 0.85  # Fluidity
    assert original_bird[1] == 0.90  # Rise

    # 2. Process with a heavy / stone description (this should induce high friction and mismatch)
    res_stone = gear.process_and_align_concept(
        concept_key="bird",
        world_description="This bird is unusually heavy, made of stone and pulled down by extreme gravity",
        raw_stimulus=b"\x01\x02\x03\x04"
    )

    assert "concept_key" in res_stone
    assert res_stone["concept_key"] == "bird"
    assert "pred_fact_distance" in res_stone
    assert "cause_fact_distance" in res_stone
    assert "tuning_rate" in res_stone
    assert "narrative" in res_stone

    # Ensure memory was modified towards the world_vector (tuning in action)
    adjusted_bird = gear.internal_cause_registry["bird"]
    # Due to the "stone/heavy" trigger words, the fluidity and rise should decrease
    assert adjusted_bird[0] < original_bird[0]
    assert adjusted_bird[1] < original_bird[1]

    # Verify that the tuning rate is proportional to mismatch
    assert res_stone["tuning_rate"] > 0.0


def test_conceptual_causal_gear_unseen_concept():
    """
    Verifies that ConceptualCausalGear dynamically seeds unseen words
    as new causes and aligns them without errors.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)
    gear = ConceptualCausalGear(mc)

    concept = "mystical_phoenix"
    assert concept not in gear.internal_cause_registry

    # Process
    res = gear.process_and_align_concept(
        concept_key=concept,
        world_description="A mystical phoenix flying high with brilliant golden wings",
        raw_stimulus=b"spark"
    )

    # New concept should now be registered and tuned
    assert concept in gear.internal_cause_registry
    assert len(gear.internal_cause_registry[concept]) == 4
