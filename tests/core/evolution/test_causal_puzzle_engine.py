"""
Test Suite for Causal Puzzle Assembly & Meta-Lensification Engine
================================================================
Verifies correct bottom-up node matching, recombination, reality feedback, and top-down lensification.
"""

import os
import pytest
import numpy as np
from core.evolution.causal_puzzle_engine import CausalPuzzleNode, CausalPuzzleRecombinationEngine
from core.memory.causal_controller import CausalMemoryController


def test_causal_puzzle_node_matching():
    """
    Verifies that compatible groves and ridges match successfully,
    while incompatible ones are rejected.
    """
    # Wing node: has groove for thrust, produces lift
    wing = CausalPuzzleNode(
        name="wing",
        grooves={"thrust_socket": np.array([0.9, 0.1])},
        ridges={"lift_power": np.array([0.8, 0.8])}
    )

    # Thrust node: has groove for fuel, produces thrust
    thrust = CausalPuzzleNode(
        name="thrust",
        grooves={"fuel_socket": np.array([0.5, 0.5])},
        ridges={"thrust_output": np.array([0.95, 0.05])}
    )

    # Rock node: has no compatible ridges/grooves
    rock = CausalPuzzleNode(
        name="rock",
        grooves={"none": np.array([0.0, 0.0])},
        ridges={"none": np.array([0.0, 0.0])}
    )

    # Wing should fit with thrust
    fits, score, g_key, r_key = wing.fits_with(thrust)
    assert fits is True
    assert score > 0.8
    assert g_key == "thrust_socket"
    assert r_key == "thrust_output"

    # Rock should NOT fit with wing
    fits_rock, _, _, _ = wing.fits_with(rock)
    assert fits_rock is False


def test_recombination_engine_flow():
    """
    Tests the full bottom-up recombination, reality feedback,
    and meta-lensification flow.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
    mc = CausalMemoryController(data_dir=data_dir)
    engine = CausalPuzzleRecombinationEngine(memory_controller=mc)

    # 1. Trigger recombination between wing and thrust (built-in default nodes)
    res = engine.trigger_recombination("wing", "thrust")
    assert res["success"] is True
    assert "wing" in res["chain"]
    assert "thrust" in res["chain"]

    # 2. Apply matching reality feedback (crystallization match)
    match_reality = {
        "reality_vector": np.array([0.90, 0.85, 0.05], dtype=np.float32)
    }
    feedback_match = engine.apply_reality_feedback(res["chain"], match_reality)
    assert feedback_match["status"] == "CRYSTALLIZED"
    assert feedback_match["error"] < 0.45

    # 3. Evaluate Meta-Lensification (Top-Down lens creation)
    lens_res = engine.evaluate_meta_lensification()
    assert lens_res is not None
    assert "CAUSAL_LENS_" in lens_res["name"]
    assert "math" in lens_res["refraction_matrix"]

    # 4. Apply mismatching reality feedback (dismantling)
    mismatch_reality = {
        "reality_vector": np.array([0.01, 0.02, 0.99], dtype=np.float32)
    }
    # Create another chain to dismantle
    res_mismatch = engine.trigger_recombination("wind", "wing")
    assert res_mismatch["success"] is True

    feedback_mismatch = engine.apply_reality_feedback(res_mismatch["chain"], mismatch_reality)
    assert feedback_mismatch["status"] == "DISMANTLED"
