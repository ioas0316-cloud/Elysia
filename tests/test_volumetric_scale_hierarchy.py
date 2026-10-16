"""
Unit tests for Volumetric Phase Sphere & Scale Hierarchy Coupling
"""

import numpy as np
import pytest
from core.physics.multiscale_volumetric_coupler import MultiScaleVolumetricPhaseCoupler, VolumetricPhaseState
from elysia_core.unified_system_architecture import ComplexPhasor, DynamicMacroRotorStack
from core.physics.causal_field import CausalField, InformationVoxel


def test_solid_angle_calculation():
    coupler = MultiScaleVolumetricPhaseCoupler()
    # At origin
    sa_origin = coupler.calculate_solid_angle([0, 0, 0], [0, 0, 0])
    assert np.isclose(sa_origin, 4.0 * np.pi)

    # At distance r = 2
    sa_dist = coupler.calculate_solid_angle([0, 0, 0], [2, 0, 0], area=1.0)
    assert np.isclose(sa_dist, 0.25)


def test_toroidal_velocity_closed_loop():
    coupler = MultiScaleVolumetricPhaseCoupler()
    v = coupler.calculate_toroidal_velocity(np.array([1.0, 1.0, 1.0], dtype=np.float32), strength=1.0)
    assert len(v) == 3
    assert np.linalg.norm(v) > 0


def test_micro_to_meso_coupling():
    coupler = MultiScaleVolumetricPhaseCoupler()
    micro_phasors = [
        ComplexPhasor(1.0, 0.0, 10.0),
        ComplexPhasor(1.0, np.pi / 2, 10.0)
    ]
    positions = [np.array([1, 0, 0], dtype=np.float32), np.array([0, 1, 0], dtype=np.float32)]

    meso_state = coupler.micro_to_meso_coupling(micro_phasors, positions)
    assert isinstance(meso_state, VolumetricPhaseState)
    assert len(meso_state.quaternion_q) == 4
    assert np.isclose(np.linalg.norm(meso_state.quaternion_q), 1.0)


def test_s3_quaternion_in_causal_field():
    cf = CausalField(dimensions=3)
    v1 = InformationVoxel("v1", "Test", np.array([1, 0, 0], dtype=np.float32), position=np.array([1, 1, 1], dtype=np.float32), velocity=np.array([0.5, 0.1, -0.2], dtype=np.float32))
    cf.add_voxel(v1)
    cf.step(0.1)

    assert len(v1.quaternion_q) == 4
    assert np.isclose(np.linalg.norm(v1.quaternion_q), 1.0)
    assert v1.solid_angle_sr <= 4.0 * np.pi
