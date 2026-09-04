"""
Unit tests for MultiGravitationalFieldInterference and Fused Causal Deformation
"""

import numpy as np
import pytest
from core.topology.multi_gravitational_field import MultiGravitationalFieldInterference
from core.topology.fused_causal_deformation import fused_causal_deformation_update, HAS_CPP_EXTENSION


def test_multi_gravitational_field_interference():
    engine = MultiGravitationalFieldInterference(dim=4)
    human_center = np.array([1.0, 0.0, 0.0, 0.0])
    elysia_center = np.array([0.0, 1.0, 0.0, 0.0])
    current_state = np.array([0.5, 0.5, 0.0, 0.0])

    res = engine.compute_interference_pattern(
        human_gravitational_center=human_center,
        elysia_gravitational_center=elysia_center,
        current_state_vector=current_state
    )

    assert "axis_distance" in res
    assert "interference_intensity" in res
    assert "composite_attractor_center" in res
    assert res["interference_intensity"] > 0.0
    assert len(res["composite_attractor_center"]) == 4


def test_fused_causal_deformation():
    # out_dim=2, in_dim=3 -> S has size out_dim=2
    S = np.zeros(2, dtype=float)
    C = np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3]], dtype=float)
    W_back = np.random.randn(3, 2)
    intent_I = np.array([1.0, 0.5, -0.2], dtype=float)

    S_up, C_up, fric = fused_causal_deformation_update(
        S=S, C=C, W_back=W_back, intent_I=intent_I, relaxation_steps=3
    )

    assert len(S_up) == 2
    assert C_up.shape == (2, 3)
    assert fric >= 0.0
