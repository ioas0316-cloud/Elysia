"""Unit tests for Enactive Boundary Phase-Lock Engine."""

import pytest
import numpy as np
from core.physics.enactive_boundary_phase_lock import (
    InterfaceBoundaryObserver,
    DynamicConstraintMapper,
    PhaseLockSynchronizer,
    EnactiveBoundaryEngine,
)


def test_interface_boundary_observer():
    observer = InterfaceBoundaryObserver(internal_capacity=100.0, internal_reach=10.0)
    protocol_spec = {
        "max_frequency": 60.0,
        "max_power": 40.0,
        "max_bandwidth": 500.0,
        "latency_tolerance": 0.02,
    }
    limits = observer.observe_external_protocol(protocol_spec)
    assert limits["max_frequency"] == 60.0
    assert limits["max_power"] == 40.0

    state = np.array([3.0, 4.0, 0.0])  # magnitude = 5.0
    clearance = observer.compute_boundary_clearance(state, limits)
    assert clearance["power_clearance"] == 35.0  # 40 - 5
    assert clearance["reach_margin"] == 5.0  # 10 - 5
    assert 0.0 <= clearance["normalized_safety_ratio"] <= 1.0


def test_dynamic_constraint_mapper_friction():
    mapper = DynamicConstraintMapper(learning_rate=0.2, decay_rate=0.05)
    initial_power = mapper.learned_boundaries["power_limit"]

    # High friction test -> boundary tightening
    high_friction = {"latency": 0.5, "error_rate": 0.1, "resistance": 0.2}
    res_high = mapper.update_from_friction(high_friction)
    assert res_high["instant_friction"] > 1.0
    assert res_high["effective_power_limit"] < initial_power

    # Low friction test -> boundary relaxation
    current_power = res_high["effective_power_limit"]
    low_friction = {"latency": 0.001, "error_rate": 0.0, "resistance": 0.0}
    res_low = mapper.update_from_friction(low_friction)
    assert res_low["effective_power_limit"] > current_power


def test_phase_lock_synchronizer():
    synchro = PhaseLockSynchronizer(internal_freq=10.0, coupling_strength=1.0)
    ext_phase = np.pi / 4.0
    ext_freq = 10.0
    dt = 0.01

    phase, diff, R = synchro.synchronize(ext_phase, ext_freq, dt, max_allowed_freq=100.0)
    assert 0.0 <= phase <= 2 * np.pi
    assert 0.0 <= R <= 1.0

    # Test frequency clamping under strict boundary limit
    _, _, R_clamped = synchro.synchronize(ext_phase, external_freq=200.0, dt=dt, max_allowed_freq=50.0)
    assert R_clamped >= 0.0


def test_enactive_boundary_engine_full_loop():
    engine = EnactiveBoundaryEngine(
        internal_capacity=100.0,
        internal_reach=10.0,
        base_freq=10.0,
        coupling_strength=0.8,
    )

    state = np.array([1.0, 2.0, 2.0])
    proto_spec = {"max_frequency": 50.0, "max_power": 30.0}
    ext_freq = 10.0
    friction = {"latency": 0.01, "error_rate": 0.01, "resistance": 0.01}

    # Step multiple times with evolving external phase to observe phase locking synchronization
    ext_phase = 0.0
    dt = 0.01
    for _ in range(100):
        ext_phase = (ext_phase + ext_freq * dt) % (2 * np.pi)
        step_res = engine.step(
            active_state=state,
            protocol_spec=proto_spec,
            external_phase=ext_phase,
            external_freq=ext_freq,
            friction_feedback=friction,
            dt=dt,
        )

    assert "order_parameter_R" in step_res
    assert "resonance_energy" in step_res
    assert step_res["order_parameter_R"] > 0.5
