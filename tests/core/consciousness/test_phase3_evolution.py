import os
import pytest
import numpy as np
from core.consciousness.autonomous_loop import ConsciousnessLoop
from core.memory.causal_controller import CausalMemoryController
from core.evolution.axis_sprouting import DynamicAxisSprouter
from core.evolution.experience_tying import ContinuousExperienceTyer

def test_phase3_evolution_dynamic_adaptation():
    """
    Verifies that Axis Sprouting and Experience Tying run flawlessly
    inside the ConsciousnessLoop.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)

    # Initialize Loop
    loop = ConsciousnessLoop(corpus_path=data_dir, memory_controller=mc, data_dir=data_dir)

    # Verify sprouter and tyer exist
    assert hasattr(loop, "axis_sprouter")
    assert hasattr(loop, "experience_tyer")

    # Run 3 cycles to trigger the new gears
    for _ in range(3):
        result = loop.process_life_cycle()
        assert "cycle" in result

    # Check that EMBODIED_SENSATION_TYING engrams are stored in Memory Controller
    tying_engrams = [
        info for info in mc.index.values()
        if info.get("data_blob", {}).get("type") == "EMBODIED_SENSATION_TYING"
    ]
    assert len(tying_engrams) > 0

    # Verify structure of tying engram
    first_tie = tying_engrams[0]["data_blob"]
    assert "concept_name" in first_tie
    assert "associated_concept" in first_tie
    assert "physical_sensation_vector" in first_tie
    assert "embodied_metaphor" in first_tie

    # Let's manually trigger sprouting to check correctness
    sprouter = DynamicAxisSprouter(mc)
    sameness_mock = {
        "sameness_variance": 0.12,
        "min_difference": 0.2,
        "best_sameness_axis": [0.1] * 12
    }
    sprout_res = sprouter.evaluate_and_sprout("Red_Color", "Jesus_Passion", sameness_mock)
    assert sprout_res is not None
    assert "axis_sprouted_Red_Color_Jesus_Passion" in sprout_res["axis_name"]

    # Check that SPROUTED_COGNITIVE_AXIS engrams are logged
    axis_engrams = [
        info for info in mc.index.values()
        if info.get("data_blob", {}).get("type") == "SPROUTED_COGNITIVE_AXIS"
    ]
    assert len(axis_engrams) > 0
