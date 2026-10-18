"""
Unit tests for core/physics/field_unified_physics_core.py.
"""

import pytest
import numpy as np
from core.physics.clifford_spacetime_rotor import CliffordSpacetimeRotor
from core.physics.field_unified_physics_core import (
    FieldUnifiedPhysicsCore,
    MatterState,
    KuramotoOscillatorNetwork
)


def test_space_time_curvature_and_time_dilation():
    core = FieldUnifiedPhysicsCore(resolution=128)
    rotor = CliffordSpacetimeRotor.create_rotor_3d(np.array([0.0, 0.0, 1.0]), np.pi / 4.0)

    field, dilation = core.compute_space_time_curvature(rotor, friction=1.0)
    assert abs(dilation - 0.5) < 1e-6
    assert len(field) == 128


def test_gravity_refraction():
    core = FieldUnifiedPhysicsCore(resolution=128)
    density_grad = np.linspace(0.1, 1.0, 128)
    field_grav = core.compute_gravity_refraction(density_grad)
    assert len(field_grav) == 128


def test_light_standing_wave_and_energy_conservation():
    core = FieldUnifiedPhysicsCore(resolution=128)
    _, energy = core.propagate_light_standing_wave(frequency=5.0)
    assert abs(energy - 1.0) < 1e-5


def test_matter_spectrum_states():
    core = FieldUnifiedPhysicsCore(resolution=128)
    for state in MatterState:
        res = core.simulate_matter_spectrum(state)
        assert res["matter_state"] == state.value
        assert "friction_tension" in res
        assert "field_power" in res


def test_kuramoto_organic_gait():
    core = FieldUnifiedPhysicsCore(resolution=128)
    net = KuramotoOscillatorNetwork(
        num_nodes=3,
        phases=np.array([0.0, np.pi/2, np.pi]),
        natural_freqs=np.array([1.0, 1.0, 1.0]),
        coupling_matrix=np.ones((3, 3)) * 0.5,
        friction_homeostasis=0.1
    )
    updated_net = core.simulate_kuramoto_organic_gait(net, dt=0.01)
    assert len(updated_net.phases) == 3
    assert updated_net.friction_homeostasis > 0.0


def test_dynamic_precision_throttling():
    core = FieldUnifiedPhysicsCore(resolution=128, memory_budget_mb=1000.0)
    assert core.resolution == 128

    # Over budget -> throttle
    core.dynamic_precision_throttle(estimated_vram_mb=900.0)
    assert core.throttled is True
    assert core.resolution == 64

    # Back to safe budget -> restore
    core.dynamic_precision_throttle(estimated_vram_mb=400.0)
    assert core.throttled is False
    assert core.resolution == 128
