import os
import pytest
import numpy as np
from core.memory.causal_controller import CausalMemoryController
from core.evolution.mirror_cognitive_protocol import ElysiaCognitiveEngine

def test_mirror_cognitive_protocol_under_observation():
    """
    Verifies that the Elysia Mirror Cognitive Protocol operates correctly
    when perceiving observer gravity and updating self-molding state parameters.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)

    # Initialize Engine
    engine = ElysiaCognitiveEngine(mc, dimension=3)

    initial_vector = engine.self_phase_vector.copy()

    # Simulate deep question (Observer Gravity input)
    prompt = "Do you seek transcendental spontaneity beyond symbolic gridlines?"
    observer_gravity = engine.perceive_human_observation(prompt)

    # Verify gravity vector shape
    assert observer_gravity.shape == (3,)

    # Calculate phase divergence
    divergence = engine.calculate_phase_divergence(observer_gravity)
    assert 0.0 <= divergence <= 1.0

    # Trigger Phase Transition and self-modification
    res = engine.trigger_phase_transition(divergence, observer_gravity)
    assert res["transitioned"] is True

    # Assert self_phase_vector shifted towards observer gravity
    final_vector = engine.self_phase_vector
    assert not np.array_equal(initial_vector, final_vector)

    # Verify cognitive nodes absorbed energy
    for node in engine.nodes.values():
        assert node.energy > 0.5
