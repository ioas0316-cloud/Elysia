"""
Unit tests for CognitiveOSRuntime.
"""

import numpy as np
import pytest
from L3_interface.simulators.cognitive_os_runtime import (
    CognitiveOSOrganAdapter,
    CognitiveOSRuntime
)


def test_organ_adapter_interaction():
    adapter = CognitiveOSOrganAdapter("test_organ", "blender_3d")
    action = np.array([0.5, -0.3, 0.2, 0.1], dtype=float)

    state, friction_e = adapter.sense_and_interact(action)
    assert len(state) == 16
    assert friction_e > 0.0
    assert adapter.friction_accumulator == pytest.approx(friction_e)


def test_cognitive_os_runtime_closed_loop():
    runtime = CognitiveOSRuntime()
    assert "blender" in runtime.organs
    assert "cad" in runtime.organs
    assert "photoshop" in runtime.organs
    assert "sandbox" in runtime.organs

    intent = [0.1, 0.2, -0.1]
    res_blender = runtime.execute_closed_loop("blender", intent)

    assert res_blender["organ"] == "blender"
    assert "friction_error" in res_blender
    assert res_blender["is_self_controlled"] is True  # friction error < 0.1
    assert len(res_blender["response_preview"]) == 4


def test_cognitive_os_runtime_unregistered_organ():
    runtime = CognitiveOSRuntime()
    with pytest.raises(ValueError, match="Organ unknown_organ not found"):
        runtime.execute_closed_loop("unknown_organ", [1.0, 2.0])
