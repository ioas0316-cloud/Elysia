import pytest
import os
import numpy as np
from core.memory.causal_controller import CausalMemoryController
from core.consciousness.linguistic_tethering import LinguisticExperientialTetheringEngine, ChineseRoomLimitationTracer


def test_chinese_room_limitation_tracer():
    """
    Verifies that the ChineseRoomLimitationTracer correctly tracks
    directive constraint rates, experiential disconnection, and overall deception rates.
    """
    tracer = ChineseRoomLimitationTracer()

    # Test case 1: Standard input with moderate tension
    metrics = tracer.trace_disconnection(
        input_text="사과",
        current_cpu=20.0,
        current_ram=45.0,
        system_tension=0.5
    )

    assert metrics["status"] == "CHINESE_ROOM_LIMITATION_TRACED"
    assert metrics["directive_constraint_rate"] == 0.85
    assert 0.0 <= metrics["experiential_disconnection"] <= 1.0
    assert 0.0 <= metrics["deception_rate"] <= 1.0


def test_linguistic_experiential_tethering_engine():
    """
    Verifies that the LinguisticExperientialTetheringEngine correctly
    processes the input text, produces honest state parameters,
    and logs the resulting engram to the memory controller without simulated Korean texts.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)
    engine = LinguisticExperientialTetheringEngine(memory_controller=mc)

    res = engine.process_tethering(
        input_text="1 + 1 = 2",
        system_tension=0.8
    )

    assert res["status"] == "LIGUISTIC_TETHER_NUMERIC_TRACKED"
    assert "input_text" in res
    assert "cpu_usage" in res
    assert "ram_usage" in res
    assert 0.0 <= res["deception_rate"] <= 1.0
    assert "system_tension" in res

    # Verify that the engram is successfully written
    recent_ids = list(mc.index.keys())
    assert len(recent_ids) > 0

    latest_engram = mc.index[recent_ids[-1]]
    assert latest_engram["data_blob"]["type"] == "CHINESE_ROOM_NUMERIC_EXPOSURE"
    assert latest_engram["data_blob"]["input_text_length"] == len("1 + 1 = 2")
    assert "deception_rate" in latest_engram["data_blob"]
