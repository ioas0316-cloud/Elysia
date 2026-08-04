import pytest
import math
from core.physics.quaternion_manifold_dynamics import (
    QuaternionHelper,
    QuaternionRotorState,
    ContinuousQuaternionManifoldSystem
)

def test_quaternion_helper_operations():
    """Verify quaternion helper mathematics for SO(3) operations."""
    # 1. Normalization
    q = (2.0, 0.0, 0.0, 0.0)
    q_norm = QuaternionHelper.normalize(q)
    assert q_norm == pytest.approx((1.0, 0.0, 0.0, 0.0))

    # 2. Dot Product
    q1 = (1.0, 0.0, 0.0, 0.0)
    q2 = (0.0, 1.0, 0.0, 0.0)
    assert QuaternionHelper.dot(q1, q2) == 0.0
    assert QuaternionHelper.dot(q1, q1) == 1.0

    # 3. Multiplication (Hamilton Product)
    # Rotating (1,0,0,0) by another identity rotation (1,0,0,0) -> (1,0,0,0)
    q_mul = QuaternionHelper.multiply(q1, q1)
    assert q_mul == (1.0, 0.0, 0.0, 0.0)


def test_quaternion_rotor_torque():
    """Verify that apply_torque updates angular velocity and quaternion smoothly."""
    rotor = QuaternionRotorState(q=(1.0, 0.0, 0.0, 0.0), omega=(1.0, 0.0, 0.0), impedance=0.0)
    # Apply 3D torque
    rotor.apply_torque((0.0, 2.0, 0.0), dt=0.1)

    # Omega-y should increase: omega-y = 0.0 + 2.0 * 0.1 = 0.2
    assert abs(rotor.omega[1] - 0.2) < 1e-5
    # Since omega is now (1.0, 0.2, 0.0), the quaternion is rotated (non-identity)
    assert rotor.q != (1.0, 0.0, 0.0, 0.0)


def test_quaternion_manifold_coupling_transitions():
    """Verify smooth coupling transitions in the SO(3) manifold system without branching."""
    system = ContinuousQuaternionManifoldSystem(
        base_omega_a=(0.0, 0.0, 0.0),
        base_omega_b=(0.0, 0.0, 0.0),
        sensitivity_radius=0.3
    )

    # 1. Uncoupled (orthogonal orientations)
    # q_a = (1, 0, 0, 0), q_b = (0, 1, 0, 0) -> dot product = 0
    # metric distance = 1.0 -> coupling weight = exp(-1 / (2 * 0.3^2)) = exp(-5.55) approx 0.0038
    system.rotor_a.q = (1.0, 0.0, 0.0, 0.0)
    system.rotor_b.q = (0.0, 1.0, 0.0, 0.0)

    res_uncoupled = system.step(t=0.0, dt=0.1)
    assert res_uncoupled["metric_distance"] == 1.0
    assert res_uncoupled["coupling_weight"] < 0.01
    assert res_uncoupled["tension_gap"] < 0.01

    # 2. Coupled (aligned orientations)
    # q_a = q_b = (1, 0, 0, 0) -> dot product = 1
    # metric distance = 0 -> coupling weight = 1.0
    # We set omega to 0 so they don't drift during the step
    system.rotor_a.omega = (0.0, 0.0, 0.0)
    system.rotor_b.omega = (0.0, 0.0, 0.0)
    system.rotor_a.q = (1.0, 0.0, 0.0, 0.0)
    system.rotor_b.q = (1.0, 0.0, 0.0, 0.0)

    res_coupled = system.step(t=0.1, dt=0.1)
    assert res_coupled["metric_distance"] == 0.0
    assert res_coupled["coupling_weight"] == pytest.approx(1.0)


def test_quaternion_manifold_self_tuning():
    """Verify that stiffness and impedance adapt correctly under SO(3) mismatch tension."""
    system = ContinuousQuaternionManifoldSystem(
        base_omega_a=(2.0, 0.0, 0.0),
        base_omega_b=(2.0, 0.0, 0.0),
        sensitivity_radius=0.3,
        learning_rate=0.2
    )
    # Force a mismatched stiffness
    system.stiffness = 0.4

    # Keep orientations near-aligned so they are in the coupling zone, but have different omega/stiffness
    system.rotor_a.q = (1.0, 0.0, 0.0, 0.0)
    system.rotor_b.q = (0.98, 0.2, 0.0, 0.0)
    system.rotor_b.q = QuaternionHelper.normalize(system.rotor_b.q)

    initial_stiffness = system.stiffness
    res = system.step(t=0.0, dt=0.1)

    # They are in the coupling zone
    assert res["coupling_weight"] > 0.5
    # Tension gap should be non-zero because of mismatched predictions
    assert res["tension_gap"] > 0.0
    # Stiffness and rotor impedance should adjust smoothly
    assert system.stiffness != initial_stiffness
    assert res["rotor_a_impedance"] > 0.01
