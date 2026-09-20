"""
Unit tests for Phase 4: Observational Engine & Closed-Loop Pipeline
"""

import pytest
import numpy as np
from synaptic_architecture.observational_engine import ObservationalEngine


def test_observational_engine_end_to_end():
    engine = ObservationalEngine(num_nodes=64)

    # 1. Ingest code
    code = """
x = 5
a = x + 10
b = a * 5
c = b - a
"""
    graph = engine.ingesting_code_or_symbols(code)
    assert len(graph.nodes) > 0

    # 2. Observe and Adapt closed loop
    res = engine.observe_and_adapt(steps=60, interval_us=250.0)

    assert res["processed_events"] == 60
    assert res["locked_events"] > 30
    assert len(engine.phase_diff_history) > 0


def test_signal_stream_observational_loop():
    engine = ObservationalEngine(num_nodes=32)

    signals = np.sin(np.linspace(0, 4 * np.pi, 50))
    graph = engine.ingesting_signal_stream(signals, threshold=0.1)
    assert len(graph.nodes) > 0

    res = engine.observe_and_adapt(steps=30, interval_us=200.0)
    assert res["processed_events"] == 30


if __name__ == "__main__":
    test_observational_engine_end_to_end()
    test_signal_stream_observational_loop()
    print("All ObservationalEngine tests passed!")
