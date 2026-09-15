"""
Unit Tests for Predatory Boundary Expansion Engine & World Tree Network
======================================================================
"""

import pytest
import numpy as np
from core.physics.causal_field import CausalField, InformationVoxel
from core.evolution.predatory_boundary_expansion import PredatoryBoundaryExpansionEngine, CognitivePlateTectonics
from core.evolution.world_tree_network import WorldTreeNetwork, WorldTreeNodeInstance


def test_predatory_boundary_expansion_basic():
    causal_field = CausalField()
    engine = PredatoryBoundaryExpansionEngine(causal_field=causal_field, stress_threshold=1.0)

    p1 = InformationVoxel("v1", "Predator", np.array([0.7, 0.3, 0.0], dtype=np.float32), mass=1.0, position=np.array([0.0, 0.0, 0.0]))
    p2 = InformationVoxel("v2", "Prey", np.array([0.0, 0.0, 1.0], dtype=np.float32), mass=1.0, position=np.array([0.1, 0.0, 0.0]))

    causal_field.add_voxel(p1)
    causal_field.add_voxel(p2)

    res = engine.execute_predatory_interaction("v1", "v2", assimilation_ratio=0.5)

    assert res["predator_new_mass"] == 1.5
    assert p2.mass == 0.5
    assert res["friction_intensity"] > 0
    assert len(causal_field.beams) == 1


def test_cognitive_plate_tectonics_uplift():
    tectonics = CognitivePlateTectonics(stress_threshold=1.0)

    triggered, data = tectonics.accumulate_friction(0.5)
    assert not triggered

    triggered, data = tectonics.accumulate_friction(0.6)
    assert triggered
    assert data["phase_transition"] == "TECTONIC_RUPTURE_UPLIFT"
    assert len(tectonics.uplift_history) == 1


def test_world_tree_network_bootstrap_and_sap():
    tree = WorldTreeNetwork()

    assert len(tree.nodes) == 3
    assert "node_root" in tree.nodes
    assert "node_trunk" in tree.nodes
    assert "node_canopy" in tree.nodes

    tree.nodes["node_root"].receive_friction(1.0)
    report = tree.circulate_sap_flow(dt=0.1)

    assert report["total_friction_processed"] > 0
    assert tree.community_wisdom_level > 1.0


def test_world_tree_sacrificial_apoptosis():
    tree = WorldTreeNetwork()
    initial_sap = tree.sap_reservoir

    res = tree.handle_sacrificial_node("node_root")
    assert res["success"]
    assert tree.nodes["node_root"].sacrificed
    assert not tree.nodes["node_root"].is_active
    assert tree.sap_reservoir > initial_sap


def test_generational_sprouting_preserving_s_abs():
    tree = WorldTreeNetwork()

    sprout = tree.sprout_next_generation_branch("node_trunk", role="canopy_generative")
    assert sprout["success"]
    assert sprout["generation"] == 2
    assert sprout["archetype_spine_preserved"] > 0.99 # Highly aligned with S_abs

    chorus = tree.sing_forest_chorus()
    assert chorus["chorus_harmony"] > 0.99
    assert chorus["active_node_count"] == 4
