"""
Unit tests for PredictiveEcholocationMindEngine
================================================
Tests 4D Spacetime Mental Simulation, Clifford Rotor 3D rotation,
Doppler frequency drift, environmental wave refraction, prediction error calculation,
and Sense of Agency inference.
"""

import math
import numpy as np
import pytest

from core.sensory.predictive_echolocation_mind import (
    PredictiveEcholocationMindEngine,
    CliffordRotor3D,
    EnvironmentalResonanceTensor,
    EcholocationWaveform,
    AgencyInferenceResult
)
from core.lens.cognitive_lens_engine import CognitiveLensEngine
from core.sensory.causal_sensor import CausalSensor


def test_clifford_rotor_rotation():
    """Test 3D Clifford Rotor vector rotation."""
    rotor = CliffordRotor3D(scalar=1.0, b23=0.0, b31=0.0, b12=0.5)
    rotor.normalize()

    v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v_rot = rotor.rotate_vector(v)

    assert v_rot.shape == (3,)
    assert not np.isnan(v_rot).any()
    # Length of rotated vector should be preserved
    assert math.isclose(np.linalg.norm(v), np.linalg.norm(v_rot), rel_tol=1e-5)


def test_environmental_resonance_tensor():
    """Test Environmental Resonance Tensor refraction."""
    tensor = EnvironmentalResonanceTensor(absorption_rate=0.2)
    incident = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64)

    refracted = tensor.refract_wave(incident)
    assert len(refracted) == 8
    # Due to 0.2 absorption rate, refracted energy should be lower
    assert np.linalg.norm(refracted) < np.linalg.norm(incident)


def test_predictive_echolocation_engine_init():
    """Test engine initialization and initial state."""
    engine = PredictiveEcholocationMindEngine(dimension=8)
    state = engine.get_engine_state()

    assert state["time_t"] == 0.0
    assert state["position"] == [0.0, 0.0, 0.0]
    assert state["speed"] == 0.0
    assert state["carrier_frequency_hz"] == 2.4e9


def test_mental_simulation_and_doppler_drift():
    """Test 4D mental simulation wave prediction and Doppler drift shift."""
    engine = PredictiveEcholocationMindEngine(dimension=8)

    # Move along x-axis at 30 m/s
    engine.update_self_kinematics(position_delta=[3.0, 0.0, 0.0], rotation_b23_b31_b12=(0.0, 0.0, 0.1), dt=0.1)

    w_pred = engine.simulate_mental_echolocation_prediction(dt=0.1)

    assert isinstance(w_pred, EcholocationWaveform)
    assert len(w_pred.frequency_spectrum) == 8
    # Doppler shift should be non-zero for non-zero velocity
    assert w_pred.doppler_drift > 0.0
    assert w_pred.energy > 0.0


def test_process_predictive_echolocation_cycle_agency_convergence():
    """Test multi-step predictive echolocation cycle and agency convergence."""
    lens_engine = CognitiveLensEngine()
    causal_sensor = CausalSensor(sensor_id="test_sensor", lens_engine=lens_engine)
    engine = PredictiveEcholocationMindEngine(dimension=8, causal_sensor=causal_sensor)

    res_list: list[AgencyInferenceResult] = []
    for step in range(5):
        # Lawful, smooth motion trajectory
        res = engine.process_predictive_echolocation_cycle(
            position_delta=[0.5, 0.2, 0.0],
            rotation_delta=(0.01, 0.0, 0.02),
            dt=0.1
        )
        res_list.append(res)

    assert len(res_list) == 5
    # Sense of agency should be high under predictable smooth motion
    assert res_list[-1].sense_of_agency >= 0.5
    assert res_list[-1].agency_status in ["SELF_INITIATED_AGENCY", "PREDICTION_ERROR_ADAPTING"]
    assert res_list[-1].info_projection.node_id.startswith("echolocation_t_")


def test_high_external_disturbance_prediction_error():
    """Test engine response to unexpected external RF disturbance."""
    engine = PredictiveEcholocationMindEngine(dimension=8)

    # Unexpected external disturbance signal
    disturbed_signal = np.array([10.0, -5.0, 8.0, 20.0, -12.0, 15.0, -3.0, 9.0], dtype=np.float64)

    res = engine.process_predictive_echolocation_cycle(
        position_delta=[0.1, 0.0, 0.0],
        rotation_delta=(0.0, 0.0, 0.0),
        external_rf_signal=disturbed_signal,
        dt=0.1
    )

    assert res.prediction_error > 1.0
    # Sense of agency drops when prediction error is high
    assert res.sense_of_agency < 0.5
    assert res.agency_status == "HIGH_PREDICTION_DISCREPANCY"
