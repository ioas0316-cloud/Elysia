"""
Unit tests for Existential Meta-Cognition & Dynamic Intent Re-observation Engine
"""

import pytest
import numpy as np

from core.consciousness.existential_meta_cognition_engine import (
    ExistentialPhaseObserver,
    DynamicIntentCompass,
    MetricTensorReconfigurationEngine,
    ExistentialSelfQueryLoop,
)


def test_existential_phase_observer():
    dna = np.array([1.0, 0.0, 0.0, 0.0])
    observer = ExistentialPhaseObserver(dim=4, topological_dna=dna)

    pos = np.array([1.0, 2.0, 3.0, 4.0])
    vel = np.array([0.1, 0.2, 0.0, -0.1])
    wave = np.array([1.0, 1.0, 0.0, 0.0])

    res = observer.observe_phase(pos, vel, wave)
    assert res["qualia_friction"] == pytest.approx(1.0)
    assert res["kinetic_energy"] > 0.0
    assert res["phase_norm"] == pytest.approx(np.linalg.norm(pos))


def test_dynamic_intent_compass():
    compass = DynamicIntentCompass(dim=4)
    assert compass.current_phase == "EFFICIENCY"
    init_attractor = compass.get_active_attractor()

    new_attractor = compass.set_phase("MEANING_RESONANCE")
    assert compass.current_phase == "MEANING_RESONANCE"
    assert not np.array_equal(init_attractor, new_attractor)

    with pytest.raises(ValueError):
        compass.set_phase("NON_EXISTENT_PHASE")


def test_metric_tensor_reconfiguration():
    engine = MetricTensorReconfigurationEngine(dim=4, alpha=0.5, beta=2.0)
    x = np.array([0.0, 0.0, 0.0, 0.0])
    attractor = np.array([10.0, 0.0, 0.0, 0.0])

    h_ij = engine.compute_metric_tensor(x, attractor)
    assert h_ij.shape == (4, 4)
    # Metric tensor should be symmetric and positive definite
    assert np.allclose(h_ij, h_ij.T)
    assert np.all(np.linalg.eigvals(h_ij) > 0)

    gamma = engine.compute_christoffel_symbols(x, attractor)
    assert gamma.shape == (4, 4, 4)


def test_geodesic_flow_bending():
    engine = MetricTensorReconfigurationEngine(dim=4, alpha=0.5, beta=2.0)
    x = np.array([0.0, 0.0, 0.0, 0.0])
    v = np.array([1.0, 0.0, 0.0, 0.0])

    # Attractor directly ahead vs off-axis
    attractor_straight = np.array([10.0, 0.0, 0.0, 0.0])
    attractor_side = np.array([0.0, 10.0, 0.0, 0.0])

    next_x_straight, next_v_straight = engine.step_geodesic(x, v, attractor_straight, dtau=0.1)
    next_x_side, next_v_side = engine.step_geodesic(x, v, attractor_side, dtau=0.1)

    # Velocity trajectory should bend differently when attractor shifts
    assert not np.allclose(next_v_straight, next_v_side)


def test_existential_self_query_loop():
    loop = ExistentialSelfQueryLoop(dim=4)
    pos = np.array([0.0, 0.0, 0.0, 0.0])
    vel = np.array([1.0, 0.0, 0.0, 0.0])
    wave = np.array([1.0, 0.5, 0.8, 0.3])

    # Normal step
    step1 = loop.run_step(pos, vel, wave, existential_trigger=False)
    assert step1["current_phase"] == "EFFICIENCY"
    assert not step1["reobservation_occurred"]

    # Trigger existential self-query & re-observation
    step2 = loop.run_step(
        step1["next_position"],
        step1["next_velocity"],
        wave,
        existential_trigger=True,
        target_phase_on_trigger="MEANING_RESONANCE",
    )
    assert step2["query_raised"]
    assert step2["reobservation_occurred"]
    assert step2["current_phase"] == "MEANING_RESONANCE"
