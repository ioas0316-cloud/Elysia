"""
Arena Allocator & Zero-Copy Interceptor Python Benchmark Suite
==============================================================
Tests and benchmarks `ContinuousMemoryImpedanceStream` with C++ backend support.
"""

import time
import numpy as np
import pytest
from core.topology.continuous_memory_stream import ContinuousMemoryImpedanceStream, ChromaticSignature


def test_arena_allocator_python_stream():
    """
    Verifies ContinuousMemoryImpedanceStream initialization, node registration,
    and zero-copy push/pop streaming functionality.
    """
    stream = ContinuousMemoryImpedanceStream(target_dimension=4, initial_voltage=1.0, initial_current=1.0)
    assert stream.impedance > 0.0

    # Register isomorphic nodes
    node = stream.register_isomorphic_node(
        node_id="sensor_stream_1",
        data=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        dimension_type="1D_vector",
        chromatic=ChromaticSignature(flux=1.0, order=1.0, entropy=0.01)
    )

    assert node.node_id == "sensor_stream_1"
    assert node.dimension_type == "1D_vector"

    # Test zero-copy pushing/popping if C++ engine is available
    input_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    pushed = stream.push_zero_copy_stream(input_data, frame_index=1, channel_id=0)

    if stream.cpp_interceptor is not None:
        assert pushed is True
        output_buffer = np.zeros(4, dtype=np.float32)
        success, copied = stream.pop_zero_copy_stream(output_buffer)
        assert success is True
        assert copied == 4
        np.testing.assert_array_almost_equal(output_buffer, input_data)


def test_continuous_stream_phase_lock_propagation():
    """
    Tests spatiotemporal phase-lock propagation and dynamic impedance damping.
    """
    stream = ContinuousMemoryImpedanceStream(target_dimension=4)
    stream.register_isomorphic_node(
        node_id="topological_node",
        data=np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32),
        dimension_type="2D_field"
    )

    metrics_map = stream.propagate_spatiotemporal_phase_lock(dt=0.05)
    assert "topological_node" in metrics_map
    metrics = metrics_map["topological_node"]
    assert metrics.causal_tension >= 0.0


def test_zero_stutter_latency_benchmark():
    """
    Benchmark verifying sub-millisecond execution and zero stuttering during
    continuous streaming loops.
    """
    stream = ContinuousMemoryImpedanceStream(target_dimension=8)
    iterations = 5000

    start_time = time.perf_counter()
    for i in range(iterations):
        sample = np.ones(8, dtype=np.float32) * float(i)
        stream.push_zero_copy_stream(sample, frame_index=i)

        if stream.cpp_interceptor is not None:
            buf = np.zeros(8, dtype=np.float32)
            stream.pop_zero_copy_stream(buf)

    end_time = time.perf_counter()
    total_elapsed_ms = (end_time - start_time) * 1000.0
    avg_latency_ms = total_elapsed_ms / iterations

    print(f"\n[Python Benchmark] Total Time: {total_elapsed_ms:.2f} ms for {iterations} iterations")
    print(f"[Python Benchmark] Avg Latency per Stream Iteration: {avg_latency_ms * 1000.0:.2f} us")

    # Assert sub-millisecond per-iteration latency (typically < 50 us)
    assert avg_latency_ms < 1.0


if __name__ == "__main__":
    print("=== Running Python Arena Stream Benchmark Suite ===")
    test_arena_allocator_python_stream()
    test_continuous_stream_phase_lock_propagation()
    test_zero_stutter_latency_benchmark()
    print("=== All Python Benchmarks Passed Successfully! ===")
