"""
Unit and Integration Tests for Elysia Cross-Disciplinary Knowledge Expansion & Puzzle Synthesis
=============================================================================================
Verifies that Elysia can autonomously bind multi-modal concepts and expand her structured
knowledge across disciplines (Ecology and Thermodynamics).
"""

import pytest
import numpy as np

from core.sensory.experiential_language_mapper import (
    ExperientialLanguageMapper,
    PhysicalSensationProfile,
    HomeostasisDeficit,
    ExperienceType
)
from core.evolution.causal_puzzle_engine import (
    CausalPuzzleRecombinationEngine,
    CausalPuzzleNode
)
from core.evolution.media_ontology import MediaOntologyEngine


def test_hebbian_multimodal_binding():
    """Verifies that Hebbian learning correctly updates homeostatic profiles when presented with multi-sensory data."""
    mapper = ExperientialLanguageMapper()

    # Check that initial state has baseline homeostasis
    initial_love = mapper.homeostasis.love

    # We simulate a rich multi-sensory profile for the Eagle ("독수리")
    eagle_sensory = PhysicalSensationProfile(
        optical=700.0,
        acoustic=600.0,
        tactile=3.0,
        thermal=299.0,
        autonomic_pulse=0.6
    )

    # Active learning step
    mapper.acquire_word_step(
        symbol="독수리",
        active_sensation=eagle_sensory,
        active_deficit=HomeostasisDeficit(0.1, 0.4, 0.2),
        exp_type=ExperienceType.PHYSICAL,
        learning_rate=0.5
    )

    recalled = mapper.tethering.recall_symbol("독수리")
    assert recalled is not None
    assert recalled["exp_type"] == ExperienceType.PHYSICAL
    assert recalled["sensation"].optical == pytest.approx(350.0, abs=0.1) # Starts from 0.0, steps by 50% towards 700.0
    assert recalled["deficit"].love == pytest.approx(0.3, abs=0.1)      # Starts from 0.5, steps by 50% towards 0.1


def test_cross_disciplinary_sprouting_and_recombination():
    """Verifies that different disciplines (Ecology and Thermodynamics) sprout distinct sockets and recombine successfully."""
    puzzle_engine = CausalPuzzleRecombinationEngine()

    # 1. Sprout ecological concept: "독수리"
    node_eagle = puzzle_engine.sprout_dynamic_node("독수리")
    assert node_eagle is not None

    # Sprout ecological concept: "산림"
    node_forest = puzzle_engine.sprout_dynamic_node("산림")
    assert node_forest is not None

    # Add ecological grooves/ridges
    node_eagle.grooves["forest_habitat"] = np.array([0.9, 0.1, 0.1], dtype=np.float32)
    node_forest.ridges["forest_habitat"] = np.array([0.92, 0.08, 0.15], dtype=np.float32)

    # Attempt ecological assembly
    recomb_eco = puzzle_engine.trigger_recombination("독수리", "산림")
    assert recomb_eco["success"] is True
    assert recomb_eco["groove"] == "forest_habitat"
    assert recomb_eco["ridge"] == "forest_habitat"

    # 2. Sprout thermodynamics concept: "물"
    node_water = puzzle_engine.sprout_dynamic_node("물")
    # Sprout thermodynamics concept: "열원"
    node_heat = puzzle_engine.sprout_dynamic_node("열원")

    node_water.grooves["thermal_energy"] = np.array([0.1, 0.9, 0.9], dtype=np.float32)
    node_heat.ridges["thermal_energy"] = np.array([0.15, 0.88, 0.92], dtype=np.float32)

    recomb_phys = puzzle_engine.trigger_recombination("물", "열원")
    assert recomb_phys["success"] is True
    assert recomb_phys["groove"] == "thermal_energy"
    assert recomb_phys["ridge"] == "thermal_energy"


def test_crystallization_and_meta_lensification():
    """Verifies that stable puzzles are crystallized and synthesized into top-down lenses."""
    puzzle_engine = CausalPuzzleRecombinationEngine()

    node_water = puzzle_engine.sprout_dynamic_node("물")
    node_heat = puzzle_engine.sprout_dynamic_node("열원")

    node_water.grooves["thermal_energy"] = np.array([0.1, 0.9, 0.9], dtype=np.float32)
    node_heat.ridges["thermal_energy"] = np.array([0.15, 0.88, 0.92], dtype=np.float32)

    recomb = puzzle_engine.trigger_recombination("물", "열원")
    assert recomb["success"] is True

    # Match criteria
    simulated_reality = {
        "reality_vector": np.array([0.15, 0.88, 0.92], dtype=np.float32)
    }

    # Crystallize
    feedback_res = puzzle_engine.apply_reality_feedback(recomb["chain"], simulated_reality)
    assert feedback_res["status"] == "CRYSTALLIZED"
    assert "물_열원" in puzzle_engine.crystallized_chains or "열원_물" in puzzle_engine.crystallized_chains

    # Meta-lensification
    lens = puzzle_engine.evaluate_meta_lensification()
    assert lens is not None
    assert lens["name"].startswith("CAUSAL_LENS_")
    assert "math" in lens["refraction_matrix"]
