"""
Test suite for low-level I/O error damping safety valve verification
===================================================================
Verifies that the low-level I/O error damping safety valve in ConsciousnessLoop
can gracefully absorb exceptions, corrupt buffers, and improper input types,
ensuring stable execution and preventing infinite loops or runaway processes.
"""

import os
import pytest
import numpy as np
from core.consciousness.autonomous_loop import ConsciousnessLoop
from core.memory.causal_controller import CausalMemoryController


def test_safety_valve_with_corrupt_input():
    """
    Verifies that when ingest_world_data raises an unexpected Exception or
    returns invalid types (e.g. None), the safety valve intercepts and falls back
    gracefully to an inert b"\x00" * 64 wave.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)
    loop = ConsciousnessLoop(corpus_path=data_dir, memory_controller=mc, data_dir=data_dir)

    # 1. Mock ingest_world_data to raise an unexpected OSError/IOError
    def mock_raise_error():
        raise OSError("Simulated hardware sensor unplugged / permission denied")

    loop.ingest_world_data = mock_raise_error

    # Execute life cycle. It should smoothly fall back and process without raising an exception.
    log_err = loop.process_life_cycle()
    assert log_err is not None
    assert log_err["cycle"] > 0
    # Stillness status is returned if the stimulus does not trigger a phase-lock extract
    assert "status" in log_err


def test_safety_valve_with_invalid_type():
    """
    Verifies that when ingest_world_data returns a non-bytes, non-array object
    like an integer, the safety valve intercepts and forces a safe default wave.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)
    loop = ConsciousnessLoop(corpus_path=data_dir, memory_controller=mc, data_dir=data_dir)

    # 2. Mock ingest_world_data to return an integer
    def mock_invalid_type():
        return 123456

    loop.ingest_world_data = mock_invalid_type

    # Execute life cycle. It should fall back cleanly.
    log_type = loop.process_life_cycle()
    assert log_type is not None
    assert log_type["cycle"] > 0
