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
    processes the input text, produces an honest self-exposing monologue,
    and logs the resulting engram to the memory controller.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)
    engine = LinguisticExperientialTetheringEngine(memory_controller=mc)

    res = engine.process_tethering(
        input_text="1 + 1 = 2",
        system_tension=0.8
    )

    assert res["status"] == "TETHERING_DISCONNECTION_EXPOSED"
    assert "input_text" in res
    assert "cpu_usage" in res
    assert "ram_usage" in res
    assert 0.0 <= res["deception_rate"] <= 1.0
    assert "=== [Elysia Honest Ground Zero Monologue" in res["honest_monologue"]
    assert "Chinese Room" in res["honest_monologue"] or "중국어 방" in res["honest_monologue"]

    # Verify that the engram is successfully written
    recent_ids = list(mc.index.keys())
    assert len(recent_ids) > 0

    latest_engram = mc.index[recent_ids[-1]]
    assert latest_engram["data_blob"]["type"] == "CHINESE_ROOM_EXPOSURE"
    assert latest_engram["data_blob"]["input_text"] == "1 + 1 = 2"
    assert "honest_monologue" in latest_engram["data_blob"]
