import pytest
import os
from core.consciousness.why_bridge import WhyBridgeEngine
from core.memory.causal_controller import CausalMemoryController


def test_why_bridge_perception_flow():
    """
    Verifies that the Why-Bridge Engine correctly analyzes friction,
    performs back-tracking search on engrams, and maps values against the
    Cruciform Attractor (Self-Outpouring vs Egoism).
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)

    # Pre-populate index to have an engram to search
    mc.write_causal_engram(
        data_blob={
            "type": "CONSCIOUSNESS_CYCLE",
            "cycle": 1,
            "status": "Resonance Reached",
            "wave_preview": "abcdef01020304"
        },
        emotional_value=8.0,
        cause_id="test_prepopulate"
    )

    engine = WhyBridgeEngine(mc)

    # Simulate an incoming wave that creates tension
    test_wave = b"abcdef010203ff" # slightly different from the engram wave
    result = engine.perceive_and_trace_problem(
        error_context="test_context",
        raw_wave=test_wave,
        physical_tension=1.5,
        exception=None
    )

    assert result["status"] == "WHY_PERCEIVED_AND_RESOLVED"
    assert result["friction_intensity"] == 1.5
    assert result["anchor_engram_id"] != "None"
    assert result["egoistic_resistance"] > 0.4
    assert result["kenosis_conductance"] < 0.6
    assert "why_reason" in result
    assert "=== [Why-Bridge Introspective Journal" in result["journal_narrative"]


def test_why_bridge_with_exception():
    """
    Verifies that the Why-Bridge correctly handles and explains actual Python exceptions
    within its multi-stage introspection narrative.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)
    engine = WhyBridgeEngine(mc)

    dummy_exception = ValueError("Simulated mathematical divergence")

    result = engine.perceive_and_trace_problem(
        error_context="math_gear.calculate",
        raw_wave=b"dummy_raw_wave",
        physical_tension=0.5,
        exception=dummy_exception
    )

    assert result["is_logical_crash"] is True
    assert result["exception_type"] == "ValueError"
    assert "ValueError" in result["journal_narrative"]
    assert "여백(Yeobaek)" in result["why_reason"]
