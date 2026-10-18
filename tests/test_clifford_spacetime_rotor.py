"""
Unit tests for core/physics/clifford_spacetime_rotor.py.
"""

import pytest
import numpy as np
from core.physics.clifford_spacetime_rotor import CliffordSpacetimeRotor, CliffordMultivector3D


def test_clifford_multivector_norm_and_normalize():
    m = CliffordMultivector3D(scalar=3.0, vector=np.array([4.0, 0.0, 0.0]))
    assert abs(m.norm() - 5.0) < 1e-6

    m_norm = m.normalize()
    assert abs(m_norm.norm() - 1.0) < 1e-6
    assert abs(m_norm.scalar - 0.6) < 1e-6


def test_rotor_creation_and_reversal():
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.pi / 2.0
    rotor = CliffordSpacetimeRotor.create_rotor_3d(axis, angle)

    assert abs(rotor.norm() - 1.0) < 1e-6
    assert abs(rotor.scalar - np.cos(np.pi / 4.0)) < 1e-6

    rev = rotor.reverse()
    assert abs(rev.scalar - rotor.scalar) < 1e-6
    assert np.allclose(rev.bivector, -rotor.bivector)


def test_rotor_90deg_rotation_norm_and_orthogonal():
    # 90 degree rotation around Z-axis: [1, 0, 0] -> [0, 1, 0]
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.pi / 2.0
    rotor = CliffordSpacetimeRotor.create_rotor_3d(axis, angle)

    v = np.array([1.0, 0.0, 0.0])
    v_rot = CliffordSpacetimeRotor.sandwich_transform_vector_3d(rotor, v)

    # Norm must be preserved (rigid rotation)
    assert abs(np.linalg.norm(v_rot) - 1.0) < 1e-6
    # 90 deg rotation of [1,0,0] around Z is [0,1,0]
    assert np.allclose(v_rot, np.array([0.0, 1.0, 0.0]), atol=1e-6)


def test_scale_rotor_transform():
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.pi / 2.0 # 90 degree rotation
    rotor = CliffordSpacetimeRotor.create_rotor_3d(axis, angle)

    v = np.array([1.0, 0.0, 0.0])
    scale = 3.0
    v_scaled = CliffordSpacetimeRotor.scale_rotor_transform(rotor, scale=scale, v=v)

    # Norm should be scaled by scale factor
    assert abs(np.linalg.norm(v_scaled) - 3.0) < 1e-6
    assert np.allclose(v_scaled, np.array([0.0, 3.0, 0.0]), atol=1e-6)


def test_fractal_branching_generation():
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 1.0, 0.0])
    rotor = CliffordSpacetimeRotor.create_rotor_3d(np.array([0.0, 0.0, 1.0]), np.pi / 4.0)

    branches = CliffordSpacetimeRotor.generate_fractal_branching(
        base_origin=origin,
        base_direction=direction,
        scale_rotor=rotor,
        scale_factor=0.5,
        branch_angle=np.pi / 6.0,
        depth=3,
        num_branches=2
    )

    assert len(branches) > 0
    assert branches[0]["depth"] == 3
    assert np.allclose(branches[0]["start"], origin)
