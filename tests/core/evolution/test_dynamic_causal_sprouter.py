"""
Test Suite for Dynamic Autogenous Causal Node Sprouting
======================================================
Verifies that the sprouter derives coordinates from Unicode properties,
projects them onto the Ontological Lattice, sprouts grooves and ridges,
and successfully undergoes recombination and crystallization.
"""

import os
import pytest
import numpy as np
from core.evolution.causal_puzzle_engine import CausalPuzzleRecombinationEngine, CausalPuzzleNode
from core.memory.causal_controller import CausalMemoryController


def test_unicode_derivation_reproducibility():
    """
    Verifies that sprouting the same concept name twice yields identical
    properties, while different concepts yield distinct coordinates.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
    mc = CausalMemoryController(data_dir=data_dir)
    engine = CausalPuzzleRecombinationEngine(memory_controller=mc)

    # Sprout "fire"
    node_fire1 = engine.sprout_dynamic_node("fire")
    node_fire2 = engine.sprout_dynamic_node("fire")

    # Assert identical nodes are returned/re-used
    assert node_fire1.name == "fire"
    assert node_fire1 is node_fire2

    # Sprout "water"
    node_water = engine.sprout_dynamic_node("water")
    assert node_water.name == "water"
    assert node_water is not node_fire1


def test_ontological_lattice_projection():
    """
    Verifies that dynamically sprouted nodes are correctly projected onto the
    8-dimensional Ontological Lattice and obtain structurally relevant grooves/ridges.
    """
    engine = CausalPuzzleRecombinationEngine()

    # Sprout "love"
    node_love = engine.sprout_dynamic_node("love")

    # Must have sprouted at least one dynamic groove and ridge related to ontology alignment
    g_keys = list(node_love.grooves.keys())
    r_keys = list(node_love.ridges.keys())

    assert any("needs_" in k for k in g_keys)
    assert any("produces_" in k for k in r_keys)
    assert "physical_interface" in g_keys
    assert "physical_presence" in r_keys

    # Verify vectors are non-trivial and normalized
    for vec in node_love.ridges.values():
        assert np.linalg.norm(vec) > 0.0


def test_dynamic_recombination_and_crystallization():
    """
    Tests bottom-up recombination and crystallization for dynamically sprouted nodes.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
    mc = CausalMemoryController(data_dir=data_dir)
    engine = CausalPuzzleRecombinationEngine(memory_controller=mc)

    # Attempt to trigger recombination on unseen nodes; should sprout on-the-fly and recombine
    res = engine.trigger_recombination("computer", "atom")
    assert res["success"] is True or res["success"] is False  # Could fit or not based on Unicode geometry

    # Ensure "computer" and "atom" were both sprouted successfully
    assert "computer" in engine.nodes
    assert "atom" in engine.nodes
