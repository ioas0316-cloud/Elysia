"""
Tests for Continuous Memory Impedance Stream Engine
===================================================
Verifies dimensional isomorphism, inflection point detection, and dynamic impedance damping.
"""

import numpy as np
import pytest
from core.topology.continuous_memory_stream import (
    ContinuousMemoryImpedanceStream,
    ChromaticSignature,
    DimensionalIsomorphicNode
)


def test_dimensional_isomorphism():
    engine = ContinuousMemoryImpedanceStream(target_dimension=4)

    # 0D Point
    node_0d = engine.register_isomorphic_node("point_0d", 42.0)
    assert node_0d.dimension_type == "0D_point"
    assert node_0d.raw_shape in [(), (1,)]

    # 1D Vector
    vec_data = [1.0, 2.0, 3.0, 4.0]
    node_1d = engine.register_isomorphic_node("vec_1d", vec_data)
    assert node_1d.dimension_type == "1D_vector"
    assert node_1d.raw_shape == (4,)

    # 2D Field
    field_data = np.ones((4, 4), dtype=np.float32)
    node_2d = engine.register_isomorphic_node("field_2d", field_data)
    assert node_2d.dimension_type == "2D_field"
    assert node_2d.raw_shape == (4, 4)

    # 4D Spatiotemporal
    manifold_data = np.ones((2, 2, 2, 2), dtype=np.float32)
    node_4d = engine.register_isomorphic_node("manifold_4d", manifold_data)
    assert node_4d.dimension_type == "4D_spatiotemporal"
    assert node_4d.raw_shape == (2, 2, 2, 2)


def test_inflection_detection_and_impedance_damping():
    engine = ContinuousMemoryImpedanceStream(target_dimension=4, initial_voltage=2.0, initial_current=1.0)
    node = engine.register_isomorphic_node("stream_node", [1.0, 0.0, -1.0, 0.5])

    # Inject entropy perturbation
    node.chromatic.perturb(delta_entropy=0.8)

    # Propagate waves
    metrics = engine.propagate_spatiotemporal_phase_lock(dt=0.1)

    node_metrics = metrics["stream_node"]
    assert node_metrics.causal_tension >= 0.0
    # Dynamic damping should reduce entropy and damp tension
    assert node.chromatic.entropy < 0.8


def test_contextual_phase_axis_alignment():
    engine = ContinuousMemoryImpedanceStream(target_dimension=4)
    node = engine.register_isomorphic_node("node1", [1.0, 2.0, 3.0, 4.0])

    new_context = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    engine.align_contextual_phase_axis(new_context)

    assert engine.tension == 0.0
    assert np.allclose(engine.phase_lock_axis, new_context)
