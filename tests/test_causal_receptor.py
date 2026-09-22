"""
Unit tests for Phase 1: Causal Receptor & Deconstructor
"""

import pytest
import numpy as np
from synaptic_architecture.causal_receptor import CausalReceptor, AtomicCausalGraph


def test_code_deconstruction_ssa():
    receptor = CausalReceptor()
    code = """
a = x + 5
b = a * 2
c = b - x
"""
    graph = receptor.deconstruct_code(code)

    assert len(graph.nodes) > 0
    # Check that x, 5, 2 are fetched / constant nodes
    ops = [node.op for node in graph.nodes.values()]
    assert "FETCH" in ops
    assert "ADD" in ops
    assert "MUL" in ops or "MULT" in ops
    assert "SUB" in ops

    # Check causal distance matrix
    dist_matrix = graph.compute_causal_distance_matrix()
    assert dist_matrix.shape[0] == len(graph.nodes)
    assert np.all(np.diag(dist_matrix) == 0.0)


def test_signal_stream_deconstruction():
    receptor = CausalReceptor()
    signals = np.array([0.0, 0.05, 0.5, 0.52, 1.2, 1.21, 0.1], dtype=np.float32)

    graph = receptor.deconstruct_signal_stream(signals, threshold=0.2)
    assert len(graph.nodes) > 0
    ops = [node.op for node in graph.nodes.values()]
    assert all(op == "SIGNAL_EDGE" for op in ops)


if __name__ == "__main__":
    test_code_deconstruction_ssa()
    test_signal_stream_deconstruction()
    print("All CausalReceptor tests passed!")
