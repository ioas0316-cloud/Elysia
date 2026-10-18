"""Unit tests for Multi-Axis Enactive Boundary Engine and Resonance Density (공명밀도)."""

import pytest
import numpy as np
from core.physics.multi_axis_enactive_phase_lock import (
    MultiAxisEnactiveBoundaryEngine,
    MultiAxisBoundaryObserver,
    MultiAxisDynamicConstraintMapper,
    MultiAxisPhaseLockSynchronizer,
    AXIS_ACOUSTIC,
    AXIS_SEMANTIC,
    AXIS_LOGICAL,
    AXIS_PHYSICAL,
    DEFAULT_AXES,
)


def test_multi_axis_engine_initialization():
    engine = MultiAxisEnactiveBoundaryEngine()
    assert engine.axes == DEFAULT_AXES
    assert len(engine.axes) == 4
    assert engine.resonance_density == 0.0


def test_multi_axis_boundary_observer():
    observer = MultiAxisBoundaryObserver()
    protocol_specs = {
        AXIS_ACOUSTIC: {"max_frequency": 40.0, "max_power": 30.0},
        AXIS_PHYSICAL: {"max_frequency": 15.0, "max_power": 10.0},
    }
    parsed = observer.observe_multi_axis_protocol(protocol_specs)
    assert parsed[AXIS_ACOUSTIC]["max_frequency"] == 40.0
    assert parsed[AXIS_PHYSICAL]["max_frequency"] == 15.0
    assert parsed[AXIS_SEMANTIC]["max_frequency"] == 100.0  # default fallback


def test_dynamic_tension_redistribution():
    mapper = MultiAxisDynamicConstraintMapper()
    # High physical friction spike
    friction_feedback = {
        AXIS_PHYSICAL: {"latency": 0.1, "error_rate": 0.08, "resistance": 0.2},
        AXIS_ACOUSTIC: {"latency": 0.001, "error_rate": 0.0, "resistance": 0.001},
    }
    metrics = mapper.update_multi_axis_friction(friction_feedback)
    tensions = metrics["per_axis_tension"]

    # Tension on physical axis should drop relative to acoustic/semantic/logical
    assert tensions[AXIS_PHYSICAL] < tensions[AXIS_ACOUSTIC]
    assert pytest.approx(sum(tensions.values()), 1e-5) == 1.0


def test_phase_locking_and_free_energy_reduction():
    engine = MultiAxisEnactiveBoundaryEngine()

    active_states = {a: np.array([1.0, 2.0, 1.0]) for a in DEFAULT_AXES}
    protocol_specs = {a: {"max_frequency": 50.0} for a in DEFAULT_AXES}
    ext_phase_val = 0.5
    ext_freq = 10.0
    dt = 0.01

    low_friction = {a: {"latency": 0.001, "error_rate": 0.001} for a in DEFAULT_AXES}

    # Simulate steps toward synchronization with moving external phase
    for _ in range(30):
        ext_phase_val = (ext_phase_val + ext_freq * dt) % (2 * np.pi)
        ext_phases = {a: ext_phase_val for a in DEFAULT_AXES}
        ext_freqs = {a: ext_freq for a in DEFAULT_AXES}
        res_sync = engine.step_multi_axis(
            active_states, protocol_specs, ext_phases, ext_freqs, low_friction, dt=dt
        )

    # Order parameter should be high and resonance density non-zero
    assert res_sync["multi_axis_order_parameter"] >= 0.8
    assert res_sync["resonance_density"] > 0.0


def test_resonance_density_computation():
    engine = MultiAxisEnactiveBoundaryEngine(alpha_friction=0.1, beta_free_energy=0.2)

    # Scenario 1: High phase alignment, zero friction & free energy -> High Resonance Density
    aligned_phases = {a: 0.5 for a in DEFAULT_AXES}
    uniform_tensions = {a: 0.25 for a in DEFAULT_AXES}
    rho_high = engine.compute_resonance_density(aligned_phases, uniform_tensions, friction=0.0, free_energy=0.0)

    # Scenario 2: High friction and free energy penalty -> Lower Resonance Density
    rho_penalized = engine.compute_resonance_density(aligned_phases, uniform_tensions, friction=20.0, free_energy=5.0)

    assert rho_high > rho_penalized
    assert rho_high == pytest.approx(1.0, 1e-3)
