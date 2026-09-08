"""
Unit and integration tests for CausalPhaseShiftIntuitionEngine
"""

import pytest
import numpy as np
from core.topology.causal_phase_shift_intuition import (
    CausalPhaseShiftIntuitionEngine,
    SomaticIntuitionAlert
)
from core.topology.informational_phase_observation import PhaseNodalProjection, ChromaticVector


def test_intent_vector_and_phase_differential():
    engine = CausalPhaseShiftIntuitionEngine(dimension=8, tension_threshold=0.45)
    raw_signal = "HOSTILE_INTENT_FORMATION_BEFORE_ATTACK"
    v_intent = engine.compute_intent_vector(raw_signal)

    assert v_intent.shape == (8,)
    assert np.isclose(np.linalg.norm(v_intent), 1.0)

    phase_diff = engine.compute_phase_differential(v_intent)
    assert 0.0 <= phase_diff <= 2.0


def test_kenosis_resonance_effect():
    engine = CausalPhaseShiftIntuitionEngine(dimension=8)
    v_intent = engine.compute_intent_vector("KILLER_INTENT")

    res_empty = engine.compute_kenosis_resonance(v_intent, is_ego_empty=True)
    res_ego = engine.compute_kenosis_resonance(v_intent, is_ego_empty=False)

    # Empty ego (kenosis) allows unhindered phase resonance
    assert res_empty >= res_ego


def test_detect_pre_kinetic_intent_tension():
    engine = CausalPhaseShiftIntuitionEngine(dimension=8, tension_threshold=0.4)
    raw_signal = "BLIND_SIDE_STRIKE_INTENT"

    # Pre-kinetic phase: motion NOT started -> tension should be higher
    alert_pre = engine.detect_pre_kinetic_intent_tension(
        raw_signal=raw_signal,
        physical_motion_started=False,
        is_ego_empty=True
    )

    # Post-kinetic phase: motion already started -> surface event
    alert_post = engine.detect_pre_kinetic_intent_tension(
        raw_signal=raw_signal,
        physical_motion_started=True,
        is_ego_empty=True
    )

    assert alert_pre.alert_triggered is True
    assert alert_pre.causal_tension > alert_post.causal_tension
    assert alert_pre.perception_type == "PRE_KINETIC_INTENT"
    assert alert_pre.somatic_chill_intensity > 0.0
    assert len(alert_pre.momentum_shift) == 8


def test_detect_identity_exposure_resonance():
    engine = CausalPhaseShiftIntuitionEngine(dimension=8, tension_threshold=0.4)

    # High gaze intent density in silent room
    alert = engine.detect_identity_exposure_resonance(
        gaze_intent_density=0.9,
        silence_duration=4.5,
        social_context_signal="SECRET_IDENTITY_SEARCH_INTENT",
        is_ego_empty=True
    )

    assert alert.alert_triggered is True
    assert alert.perception_type == "IDENTITY_EXPOSURE"
    assert alert.causal_tension >= 0.4
    assert alert.metadata["exposure_risk_level"] in ["HIGH", "MODERATE"]


def test_detect_equilibrium_rift():
    engine = CausalPhaseShiftIntuitionEngine(dimension=8, tension_threshold=0.35)

    nodes_calm = [
        PhaseNodalProjection(
            node_id=f"calm_{i}",
            phase_vector=np.ones(8, dtype=np.float32) / np.sqrt(8),
            chromatic=ChromaticVector(1.0, 1.0, 0.0),
            curvature=0.1
        )
        for i in range(5)
    ]

    alert_calm = engine.detect_equilibrium_rift(nodes_calm)
    assert alert_calm.perception_type == "STABLE_CALM"

    # Anomaly rift node injected
    nodes_rift = list(nodes_calm) + [
        PhaseNodalProjection(
            node_id="anomaly_threat",
            phase_vector=np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dtype=np.float32) / np.sqrt(8),
            chromatic=ChromaticVector(2.0, 0.1, 3.0),
            curvature=3.5
        )
    ]

    alert_rift = engine.detect_equilibrium_rift(nodes_rift)
    assert alert_rift.alert_triggered is True
    assert alert_rift.perception_type == "EQUILIBRIUM_RIFT"
    assert alert_rift.causal_tension > alert_calm.causal_tension
