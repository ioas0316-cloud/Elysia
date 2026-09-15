"""
Unit tests for Closed-Loop Valence Field and Sensory Feedback Engine.
"""

import numpy as np
import pytest

from core.memory.closed_loop_valence import (
    ConceptAttractor,
    SensoryFeedbackEngine,
    ClosedLoopValenceField
)


def test_concept_attractor_initialization():
    att = ConceptAttractor(
        concept_id="TEST_CONCEPT",
        center_pos=np.array([10.0, 20.0, 30.0]),
        resonant_freq=100.0,
        well_depth=2.0,
        well_radius=1.5
    )
    assert att.concept_id == "TEST_CONCEPT"
    assert np.array_equal(att.center_pos, np.array([10.0, 20.0, 30.0]))
    assert att.resonant_freq == 100.0
    assert att.well_depth == 2.0
    assert att.well_radius == 1.5
    assert att.phase_velocity is None
    assert att.latent_dimensions == []


def test_predict_observation():
    engine = SensoryFeedbackEngine()
    att = ConceptAttractor(
        concept_id="TEST_CONCEPT",
        center_pos=np.array([1.0, 2.0, 3.0]),
        resonant_freq=150.0,
        well_depth=1.0,
        well_radius=1.0
    )
    pred = engine.predict_observation(att)
    # Expected: [1.0, 2.0, 3.0, 1.5]
    expected = np.array([1.0, 2.0, 3.0, 1.5])
    assert np.allclose(pred, expected)


def test_calculate_prediction_error():
    engine = SensoryFeedbackEngine()
    pred = np.array([10.0, 0.0, 0.0, 1.0])
    actual = np.array([13.0, 4.0, 0.0, 1.0])
    err_vec, err_mag = engine.calculate_prediction_error(pred, actual)

    assert np.allclose(err_vec, np.array([3.0, 4.0, 0.0, 0.0]))
    assert pytest.approx(err_mag, 1e-5) == 5.0


def test_adapt_attractor_state_position_and_depth():
    engine = SensoryFeedbackEngine(spatial_learning_rate=0.1, depth_adaptation_rate=0.05)
    att = ConceptAttractor(
        concept_id="TEST_CONCEPT",
        center_pos=np.array([100.0, 0.0, 0.0]),
        resonant_freq=100.0,
        well_depth=2.0,
        well_radius=1.0
    )
    initial_pos = att.center_pos.copy()
    actual_sensory = np.array([110.0, 5.0, 0.0, 1.0]) # Sensory value shifted in +x and +y

    metrics = engine.adapt_attractor_state(att, actual_sensory)

    # Position should shift towards actual sensory error direction
    assert att.center_pos[0] > initial_pos[0]
    assert att.center_pos[1] > initial_pos[1]
    assert metrics["prediction_error"] > 0.0
    assert "accumulated_error" in metrics


def test_phase_velocity_trajectory_adaptation():
    engine = SensoryFeedbackEngine(spatial_learning_rate=0.1)
    att = ConceptAttractor(
        concept_id="TEST_PHASE",
        center_pos=np.array([0.0, 0.0, 0.0]),
        resonant_freq=100.0,
        well_depth=1.0,
        well_radius=1.0,
        phase_velocity=np.array([0.5, 0.0, 0.0])
    )
    actual_sensory = np.array([0.0, 0.0, 0.0, 1.0, 1.5, 0.0, 0.0])
    engine.adapt_attractor_state(att, actual_sensory)

    assert att.phase_velocity is not None
    assert len(att.phase_velocity) == 3


def test_latent_dimension_spawning():
    engine = SensoryFeedbackEngine(latent_spawn_threshold=5.0)
    att = ConceptAttractor(
        concept_id="TEST_LATENT",
        center_pos=np.array([0.0, 0.0, 0.0]),
        resonant_freq=100.0,
        well_depth=1.0,
        well_radius=1.0
    )
    high_error_sensory = np.array([100.0, 100.0, 100.0, 1.0])

    # Run multiple steps with persistent high error
    spawned = False
    for _ in range(5):
        metrics = engine.adapt_attractor_state(att, high_error_sensory)
        if metrics["latent_spawned"]:
            spawned = True
            break

    assert spawned is True
    assert len(att.latent_dimensions) > 0
    assert "LATENT_AXIS_W1" in att.latent_dimensions


def test_closed_loop_valence_field():
    field = ClosedLoopValenceField()
    att1 = ConceptAttractor("CONCEPT_1", np.array([10.0, 0.0, 0.0]), 100.0, 1.5, 1.0)
    att2 = ConceptAttractor("CONCEPT_2", np.array([20.0, 5.0, 0.0]), 150.0, 2.0, 1.0)

    field.register_attractor(att1)
    field.register_attractor(att2)

    sensor_stream = {
        "CONCEPT_1": np.array([12.0, 1.0, 0.0, 1.0]),
        "CONCEPT_2": np.array([22.0, 6.0, 0.0, 1.5])
    }

    telemetry = field.process_sensorimotor_step(sensor_stream)

    assert "CONCEPT_1" in telemetry
    assert "CONCEPT_2" in telemetry
    assert telemetry["CONCEPT_1"]["prediction_error"] > 0.0
    assert telemetry["CONCEPT_2"]["prediction_error"] > 0.0
