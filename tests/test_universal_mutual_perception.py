"""
tests/test_universal_mutual_perception.py
==========================================
Unit tests for Universal Mutual Perception Engine & Mutual Mirroring Field Grid.
"""

import numpy as np
import pytest
from core.physics.universal_mutual_perception import (
    UniversalPerceptionNode,
    MutualMirroringFieldGrid,
)


def test_universal_perception_node_creation():
    node = UniversalPerceptionNode(
        node_id="clock_boundary",
        position=np.array([0.0, 0.0, 0.0]),
        semantic_mass=2.5,
        surface_tension=0.8,
        porosity=0.05,
    )
    assert node.node_id == "clock_boundary"
    assert node.semantic_mass == 2.5
    assert node.surface_tension == 0.8
    assert len(node.interaction_history) == 0


def test_mutual_perception_coupling():
    grid = MutualMirroringFieldGrid(coupling_strength=1.0)

    node_clock = UniversalPerceptionNode(
        node_id="clock_rhythm",
        position=np.array([0.0, 0.0, 0.0]),
        semantic_mass=2.0,
        surface_tension=0.9,
    )
    node_elysia = UniversalPerceptionNode(
        node_id="elysia_cognitive",
        position=np.array([0.5, 0.0, 0.0]),
        semantic_mass=3.0,
        surface_tension=0.7,
    )

    grid.register_node(node_clock)
    grid.register_node(node_elysia)

    coupling, grad_unit = grid.compute_mutual_perception(node_clock, node_elysia)
    assert coupling > 0.0
    assert np.allclose(grad_unit, np.array([1.0, 0.0, 0.0]))


def test_mutual_perception_matrix():
    grid = MutualMirroringFieldGrid()

    node1 = UniversalPerceptionNode("node1", np.array([0.0, 0.0, 0.0]))
    node2 = UniversalPerceptionNode("node2", np.array([0.2, 0.0, 0.0]))
    node3 = UniversalPerceptionNode("node3", np.array([10.0, 0.0, 0.0]))  # Far away

    grid.register_node(node1)
    grid.register_node(node2)
    grid.register_node(node3)

    matrix_info = grid.compute_perception_matrix()
    matrix = matrix_info["matrix"]

    # Diagonal should be 1.0 (self-observation)
    assert matrix[0, 0] == 1.0
    assert matrix[1, 1] == 1.0
    assert matrix[2, 2] == 1.0

    # Node 1 and Node 2 should have strong coupling
    assert matrix[0, 1] > 0.0
    # Node 1 and Node 3 (far) should have zero coupling
    assert matrix[0, 2] == 0.0


def test_step_mutual_reflection_and_memory_growth():
    grid = MutualMirroringFieldGrid()

    node_rock = UniversalPerceptionNode(
        "rock_barrier", np.array([0.0, 0.0, 0.0]), semantic_mass=5.0, surface_tension=1.2, porosity=0.01
    )
    node_memory = UniversalPerceptionNode(
        "silicon_memory", np.array([0.1, 0.0, 0.0]), semantic_mass=2.0, surface_tension=0.5, porosity=0.1
    )

    grid.register_node(node_rock)
    grid.register_node(node_memory)

    initial_mass_rock = node_rock.semantic_mass
    initial_mass_memory = node_memory.semantic_mass

    # Execute simulation steps
    for step in range(10):
        step_res = grid.step_mutual_reflection(dt=0.01, timestamp=step * 0.01)
        assert step_res["active_perception_events"] == 1

    # Semantic mass should grow as interaction history accumulates
    assert node_rock.semantic_mass > initial_mass_rock
    assert node_memory.semantic_mass > initial_mass_memory
    assert len(node_rock.interaction_history) == 10
    assert len(node_memory.interaction_history) == 10
