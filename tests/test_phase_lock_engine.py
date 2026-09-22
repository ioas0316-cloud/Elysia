"""
Unit tests for Phase 2: Phase-Lock Engine
"""

import time
import pytest
import numpy as np
from synaptic_architecture.phase_lock_engine import PhaseLockEngine, PhaseEvent


def test_phase_lock_filtering():
    engine = PhaseLockEngine(num_nodes=64, settling_window_us=(100.0, 500.0))
    received_events = []
    engine.register_callback(lambda e: received_events.append(e))

    now_ns = time.time_ns()

    # Event 1
    e1 = engine.process_raw_edge(pid=101, timestamp_ns=now_ns, edge_id=1)
    assert not e1.is_phase_locked  # First event has no delta

    # Event 2: 250us delta (within 100-500us window) -> locked
    e2 = engine.process_raw_edge(pid=101, timestamp_ns=now_ns + 250_000, edge_id=2)
    assert e2.is_phase_locked

    # Event 3: 50us delta (transient noise, outside window) -> not locked
    e3 = engine.process_raw_edge(pid=101, timestamp_ns=now_ns + 300_000, edge_id=3)
    assert not e3.is_phase_locked

    assert len(received_events) == 3


def test_simulate_edge_stream():
    engine = PhaseLockEngine(num_nodes=32)
    events = engine.simulate_edge_stream(count=20, interval_us=200.0)
    assert len(events) == 20
    # Most events with 200us +- jitter should be phase locked
    locked_count = sum(1 for e in events if e.is_phase_locked)
    assert locked_count > 10


if __name__ == "__main__":
    test_phase_lock_filtering()
    test_simulate_edge_stream()
    print("All PhaseLockEngine tests passed!")
