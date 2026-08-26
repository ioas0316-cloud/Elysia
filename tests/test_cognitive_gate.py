"""
Tests for Primitive Cognitive Gate & Recursive Stacking Verification
===================================================================
1. Discriminate test (Invariant & Variant separation)
2. Self-refinement test (Lens axis rotation driven by phase friction)
3. Recursive stacking test (Point -> Line -> Attractor crystallization)
4. Perception-Action Duality test
5. Internal Simulation test
"""

import pytest
import numpy as np
from core.topology.cognitive_gate import CognitiveGate, RecursiveCognitiveStack


def test_cognitive_gate_discriminate_and_refine():
    gate = CognitiveGate(dimension=8, eta=0.1, threshold=0.5)

    # Raw continuous signal
    signal = np.array([1.0, 0.5, -0.2, 0.8, 0.0, 0.3, -0.5, 0.1], dtype=np.float32)

    I, V = gate.discriminate(signal)

    assert len(I) == 8
    assert len(V) == 8
    # Invariant + Variant should reconstruct the original input vector
    np.testing.assert_allclose(I + V, signal, atol=1e-5)

    # Test self-refinement
    initial_S = gate.S.copy()
    friction = gate.self_refine(V)

    assert friction >= 0.0
    # S should have been updated by phase friction
    assert not np.array_equal(initial_S, gate.S)


def test_recursive_cognitive_stack_crystallization():
    stack = RecursiveCognitiveStack(layers=3, dimension=8, eta=0.1)

    # Process repeated identical signal to verify friction reduction / lens alignment
    base_signal = np.sin(np.linspace(0, np.pi, 8)).astype(np.float32)

    initial_total_friction = None
    final_total_friction = None

    for idx in range(10):
        res = stack.process_hierarchical(base_signal)
        assert len(res["layer_outputs"]) == 3
        assert len(res["top_attractor"]) == 8

        if idx == 0:
            initial_total_friction = res["total_friction"]
        if idx == 9:
            final_total_friction = res["total_friction"]

    # Friction should decrease over iterations as the stack's lenses align and crystallize
    assert final_total_friction <= initial_total_friction + 1e-5


def test_cognitive_gate_perception_action_duality():
    # Low max capacity -> triggers action under high friction
    gate = CognitiveGate(dimension=8, eta=0.2, threshold=0.3, max_capacity=0.01)

    high_friction_signal = np.random.normal(0, 2.0, 8)
    res = gate.process(high_friction_signal)

    assert res["action_triggered"] is True
    assert res["action_reprojected"] is not None
    assert len(res["action_reprojected"]) == 8


def test_cognitive_gate_internal_simulation():
    gate = CognitiveGate(dimension=8, eta=0.1, threshold=0.4)
    hypothetical_signal = np.array([2.0, 1.5, 0.0, -1.0, 0.5, 0.8, -0.4, 0.2], dtype=np.float32)

    sim_res = gate.simulate(hypothetical_signal, steps=5)

    assert "initial_friction" in sim_res
    assert "final_friction" in sim_res
    assert len(sim_res["trajectory"]) == 5
    assert sim_res["final_friction"] <= sim_res["initial_friction"]
