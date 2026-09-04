"""
Unit Tests for PhaseCompressionEngine
======================================
Tests $O(1)$ instant playback on $\\Delta P = 0$ (normal/constant terrain)
and selective minimal deformation calculation on $\\Delta P \\neq 0$ (exceptional friction),
as well as dynamic historical lens coupling (scar tensor + world events).
"""

import pytest
import numpy as np
from core.topology.phase_compression_engine import PhaseCompressionEngine, MacroPhaseVector


def test_macro_phase_vector_playback():
    points = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0]
    ])
    mpv = MacroPhaseVector(label="TestCycle", phase_points=points, cyclical_period=24.0)

    p0 = mpv.playback(0.0)
    assert np.allclose(p0, [1.0, 0.0, 0.0])

    p6 = mpv.playback(6.0)
    assert np.allclose(p6, [0.0, 1.0, 0.0])

    p12 = mpv.playback(12.0)
    assert np.allclose(p12, [-1.0, 0.0, 0.0])


def test_o1_playback_on_zero_delta_p():
    engine = PhaseCompressionEngine(dim=4, friction_threshold=0.1)

    # Time 6.0 corresponding to [0.0, 1.0, 0.0, 0.5] in Solar_Cycle_24h
    observed_intent = np.array([0.0, 1.0, 0.0, 0.5])
    res = engine.evaluate_phase_flow("Solar_Cycle_24h", time_marker=6.0, observed_intent=observed_intent)

    assert res["is_exceptional_friction"] is False
    assert res["flops_spent"] == 1
    assert "O(1)" in res["status"]


def test_exceptional_friction_awakens_deformation():
    engine = PhaseCompressionEngine(dim=4, friction_threshold=0.1)

    # Unexpected intention at time 6.0: Sun suddenly moves North/West
    unexpected_intent = np.array([2.0, -1.0, 0.5, 0.0])
    res = engine.evaluate_phase_flow("Solar_Cycle_24h", time_marker=6.0, observed_intent=unexpected_intent)

    assert res["is_exceptional_friction"] is True
    assert res["flops_spent"] > 1
    assert "Deformation Engine Awakened" in res["status"]


def test_dynamic_historical_coupling():
    engine = PhaseCompressionEngine(dim=4)

    world_event = np.array([1.0, 0.5, -0.2, 0.8])
    woodcutter_scar = np.array([0.8, 0.2, 0.5, 0.1])

    res = engine.process_dynamic_historical_coupling(
        entity_name="Woodcutter_NPC",
        world_event_vector=world_event,
        historical_scar_lens=woodcutter_scar
    )

    assert res["entity_name"] == "Woodcutter_NPC"
    assert res["potential_difference_delta_p"] > 0
    assert len(res["emergent_response_vector"]) == 4


def test_efficiency_metrics():
    engine = PhaseCompressionEngine(dim=4, friction_threshold=0.1)

    initial_calls = engine.playback_count + engine.deformation_count

    # 1 normal call
    engine.evaluate_phase_flow("Solar_Cycle_24h", time_marker=0.0, observed_intent=np.array([1.0, 0.0, 0.0, 0.1]))

    # 1 exceptional friction call
    engine.evaluate_phase_flow("Solar_Cycle_24h", time_marker=0.0, observed_intent=np.array([-5.0, 5.0, 0.0, 0.1]))

    metrics = engine.get_efficiency_metrics()
    assert metrics["total_calls"] == initial_calls + 2
    assert metrics["o1_playback_calls"] == 1
    assert metrics["active_deformation_calls"] == 1
    assert metrics["o1_playback_ratio"] == 0.5
