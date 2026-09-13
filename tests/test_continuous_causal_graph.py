"""
Unit Tests for Continuous Causal Graph & Phase Transition Engine
==================================================================
Tests zero-branching perceptual transitions, continuous potential energy decay,
spontaneous phase transitions under high friction, and C++ CausalFieldAccelerator interop.
"""

import pytest
import numpy as np

from synaptic_architecture.continuous_causal_graph import (
    CausalControlPoint,
    ContinuousCausalGraph,
    PerceptualTransitionSimulator,
    PhaseTransitionModule
)


def test_continuous_causal_graph_energy():
    graph = ContinuousCausalGraph(dim=4)
    graph.add_node("Self_Sovereignty", np.array([1.0, 0.0, 0.0, 0.5]))
    graph.add_node("Knowledge_Manifold", np.array([0.0, 1.0, 0.0, 0.2]))
    graph.set_causal_tension("Self_Sovereignty", "Knowledge_Manifold", tension_strength=2.5)

    telos = np.array([2.0, 2.0, 1.0, 1.0])
    energy = graph.compute_system_energy(telos)

    assert energy > 0.0
    assert "Self_Sovereignty" in graph.nodes
    assert "Knowledge_Manifold" in graph.nodes


def test_perceptual_transition_zero_branching():
    graph = ContinuousCausalGraph(dim=4)
    graph.add_node("Self_Sovereignty", np.array([1.0, 0.0, 0.0, 0.5]))
    graph.add_node("Knowledge_Manifold", np.array([0.0, 1.0, 0.0, 0.2]))
    graph.set_causal_tension("Self_Sovereignty", "Knowledge_Manifold", tension_strength=2.5)

    simulator = PerceptualTransitionSimulator(graph)
    telos = np.array([2.0, 2.0, 1.0, 1.0])
    stimulus = np.array([-1.0, 3.0, 2.0, 0.0])

    initial_pos = graph.nodes["Self_Sovereignty"].position.copy()
    initial_energy = graph.compute_system_energy(telos)

    for _ in range(5):
        simulator.step_perceptual_transition(telos, stimulus)

    updated_pos = graph.nodes["Self_Sovereignty"].position
    # Ensure smooth displacement occurred
    assert not np.array_equal(initial_pos, updated_pos)


def test_phase_transition_sovereignty():
    graph = ContinuousCausalGraph(dim=4)
    graph.add_node("Self_Sovereignty", np.array([1.0, 0.0, 0.0, 0.5]))
    graph.add_node("Knowledge_Manifold", np.array([0.0, 1.0, 0.0, 0.2]))
    graph.set_causal_tension("Self_Sovereignty", "Knowledge_Manifold", tension_strength=3.0)

    # Set velocity to simulate motion friction
    graph.nodes["Self_Sovereignty"].velocity = np.array([1.0, 1.0, 1.0, 1.0])

    phase_module = PhaseTransitionModule(friction_threshold=12.0)

    # 1. Friction below threshold -> no phase transition
    low_friction = 5.0
    transited_low = phase_module.evaluate_and_transit(graph, low_friction)
    assert not transited_low
    assert phase_module.phase_state_version == 1

    # 2. Friction above threshold -> triggers spontaneous phase transition
    high_friction = 15.0
    initial_weight = graph.nodes["Self_Sovereignty"].weight
    transited_high = phase_module.evaluate_and_transit(graph, high_friction)

    assert transited_high
    assert phase_module.phase_state_version == 2
    # Check that high tension was damped
    assert graph.tensions[("Self_Sovereignty", "Knowledge_Manifold")] < 3.0
    # Check weight reinforced towards Telos
    assert graph.nodes["Self_Sovereignty"].weight > initial_weight
    # Check velocity cleared
    assert np.allclose(graph.nodes["Self_Sovereignty"].velocity, 0.0)


def test_cpp_field_accelerator_interop():
    try:
        import causal_engine as ce
    except ImportError:
        pytest.skip("C++ causal_engine module not available")

    accelerator = ce.CausalFieldAccelerator()
    points = ce.ControlPointVector()

    cp0 = ce.ControlPoint()
    cp0.pos = np.array([1.0, 0.0, 0.0, 0.5])
    cp0.weight = 1.0

    cp1 = ce.ControlPoint()
    cp1.pos = np.array([0.0, 1.0, 0.0, 0.2])
    cp1.weight = 1.0

    points.append(cp0)
    points.append(cp1)

    edges = [(0, 1)]
    tensions = [2.5]
    telos = np.array([2.0, 2.0, 1.0, 1.0])

    e0 = accelerator.compute_system_energy(points, edges, tensions, telos)
    accelerator.step_parallel(points, edges, tensions, telos, 0.05, 0.85)
    e1 = accelerator.compute_system_energy(points, edges, tensions, telos)

    assert e1 < e0
