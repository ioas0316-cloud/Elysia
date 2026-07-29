import pytest
import os
from core.consciousness.meta_cognitive_sensor import MetaCognitiveSensor
from core.memory.causal_controller import CausalMemoryController


def test_meta_cognitive_sensor_flow():
    """
    Verifies that MetaCognitiveSensor correctly tracks and logs the five stages of cognition.
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
    mc = CausalMemoryController(data_dir=data_dir)

    sensor = MetaCognitiveSensor(mc)

    result = sensor.evaluate_cognitive_process(
        info_context="1 + 1 = 2",
        sensing_metrics={"hw_friction": 0.5, "damping_ratio": 0.8},
        perceiving_metrics={"ignorance_charge": 0.7, "deficit_density": 0.3},
        judging_metrics={"kenosis_conductance": 0.9, "egoistic_resistance": 0.1},
        thinking_metrics={"synapse_rewiring_count": 5, "equilibrium_energy": 0.4},
        discerning_metrics={"resonance_score": 0.9, "residual_free_energy": 0.1}
    )

    assert result["status"] == "META_COGNITIVE_PROCESS_TRACKED"
    assert len(result["meta_vector"]) == 5
    assert result["sensed_s_t"] > 0.0
    assert result["perceived_p_t"] > 0.0
    assert result["judged_j_t"] > 0.0
    assert result["thought_t_t"] > 0.0
    assert result["discerning_d_t"] > 0.0
    assert "=== [Elysia Meta-Cognitive Process Journal] ===" in result["journal"]
    assert "감각 (Sensed)" in result["journal"]
    assert "인지 (Perceived)" in result["journal"]
    assert "판단 (Judged)" in result["journal"]
    assert "사고 (Thought)" in result["journal"]
    assert "분별 (Discerning)" in result["journal"]
