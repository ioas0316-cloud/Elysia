"""
Unit tests for Arc Reactor Drive Axis Engine (`core/physics/arc_reactor_drive_axis.py`).
"""

import pytest
import numpy as np
from core.physics.arc_reactor_drive_axis import ArcReactorDriveEngine, LocalZeroFieldState
from core.physics.causal_field import CausalField, InformationVoxel


def test_inward_tension_convergence():
    """Tests that inward tension pulls voxels toward the core center."""
    engine = ArcReactorDriveEngine(dimensions=3, core_center=[0.0, 0.0, 0.0], domain_radius=2.0)

    # Position away from core
    pos_outside = np.array([3.0, 0.0, 0.0], dtype=np.float32)
    tension_force = engine.compute_inward_tension(pos_outside, mass=1.0)

    # Force vector should point toward origin (-x direction)
    assert tension_force[0] < 0.0
    assert tension_force[1] == 0.0
    assert tension_force[2] == 0.0


def test_local_zero_resistance_transition():
    """Tests continuous resistance transition from 0 inside zero domain to base resistance outside."""
    engine = ArcReactorDriveEngine(dimensions=3, domain_radius=2.0, base_resistance=1.0)

    # At core center (r = 0), resistance should be 0.0
    r_center = engine.compute_local_resistance(0.0)
    assert pytest.approx(r_center, abs=1e-5) == 0.0

    # Inside domain (r = 0.5), resistance should be very close to 0
    r_inside = engine.compute_local_resistance(0.5)
    assert r_inside < 0.05

    # Far outside domain (r = 10.0), resistance should approach base_resistance (1.0)
    r_far = engine.compute_local_resistance(10.0)
    assert pytest.approx(r_far, abs=1e-3) == 1.0


def test_phase_acceleration_and_zero_loss():
    """Tests phase acceleration under potential gradient in Local Zero domain."""
    engine = ArcReactorDriveEngine(domain_radius=2.0, flywheel_inertia=10.0, base_resistance=1.0)

    # Inside Local Zero domain (dist = 0.1), resistance is ~0
    state1 = engine.step(dt=1.0, external_potential_gradient=10.0, distance_to_core=0.1)

    # Acceleration = Torque / Inertia = 10.0 / 10.0 = 1.0 rad/s^2
    assert state1.phase_velocity > 0.9
    assert state1.energy_efficiency > 0.95


def test_inertial_flywheel_retention():
    """Tests flywheel momentum conservation when external driving torque stops in zero domain."""
    engine = ArcReactorDriveEngine(domain_radius=2.0, flywheel_inertia=10.0)

    # Drive engine up to speed
    engine.step(dt=2.0, external_potential_gradient=20.0, distance_to_core=0.0)
    speed_driven = engine.phase_velocity
    assert speed_driven > 0

    # Stop external input (gradient = 0) inside Local Zero zone
    state_coasting = engine.step(dt=1.0, external_potential_gradient=0.0, distance_to_core=0.0)

    # Speed should be virtually unchanged because resistance ~ 0
    assert pytest.approx(state_coasting.phase_velocity, rel=1e-2) == speed_driven
    assert state_coasting.flywheel_momentum > 0


def test_discerning_tension_noise_to_melody():
    """Tests turning noisy signals into phase-locked resonant melody."""
    engine = ArcReactorDriveEngine(intent_phase_frequency=2.0)

    # Create noisy signal: pure intent sine wave + heavy Gaussian noise
    t = np.linspace(0, 1.0, 100, dtype=np.float32)
    clean_sine = np.sin(2.0 * np.pi * 2.0 * t)
    noise = np.random.randn(100).astype(np.float32) * 2.0
    noisy_signal = clean_sine + noise

    harmonized_wave, harmony_score = engine.process_discerning_tension(noisy_signal, sample_rate=100.0)

    # Output signal should exist and harmony score should be in [0, 1]
    assert len(harmonized_wave) == 100
    assert 0.0 <= harmony_score <= 1.0


def test_causal_field_coupling():
    """Tests coupling ArcReactorDriveEngine with a CausalField."""
    engine = ArcReactorDriveEngine(core_center=[0.0, 0.0, 0.0], domain_radius=2.0)
    field = CausalField(dimensions=3)

    # Add a voxel inside the zero domain
    v1 = InformationVoxel("v1", "Core Voxel", np.array([1, 0, 0], dtype=np.float32), position=np.array([0.5, 0.0, 0.0], dtype=np.float32))
    field.add_voxel(v1)

    engine.step(dt=0.1, external_potential_gradient=5.0, distance_to_core=0.5)
    engine.couple_causal_field(field, dt=0.1)

    # Voxel inside zero zone should have boosted conductance and synchronized phase
    assert v1.conductance == 100.0
    assert v1.phase_angle == engine.phase_angle
