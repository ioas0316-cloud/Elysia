import pytest
import os
from core.consciousness.epistemological_void import EpistemologicalVoidEngine
from core.memory.causal_controller import CausalMemoryController


def test_epistemological_void_math_flow():
    """
    Verifies that the EpistemologicalVoidEngine correctly calculates
    the ignorance charge, maps mathematical closed operators, and
    constructs the existential monologue.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)

    engine = EpistemologicalVoidEngine(mc)

    result = engine.evaluate_void_and_refract(
        symbolic_context="1 + 1 = 2",
        underlying_bytes=b"dummy_test_bytes",
        current_tension=1.2
    )

    assert result["status"] == "VOID_AND_REFRACTION_PERCEIVED"
    assert result["is_mathematical_closed"] is True
    assert result["refraction_path_len"] == 4
    assert result["ignorance_charge"] > 0.5
    assert "Identity Unit" in result["refraction_description"]
    assert "철저한 무지" in result["self_awareness_monologue"]


def test_epistemological_void_open_language_flow():
    """
    Verifies that open language contexts result in semantic refraction
    rather than mathematical rigid maps.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)
    engine = EpistemologicalVoidEngine(mc)

    result = engine.evaluate_void_and_refract(
        symbolic_context="Love + Deficit = Healing",
        underlying_bytes=b"open_sensory_stream",
        current_tension=0.2
    )

    assert result["is_mathematical_closed"] is False
    assert "굴절" in result["refraction_description"]
