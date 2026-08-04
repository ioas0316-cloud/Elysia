import pytest
import math
from core.physics.adaptive_tensor_gear import (
    AdaptiveTensorGearSystem,
    RotorState,
    ConstantTrajectory,
    LocalTensorTheater,
    TensorOperator
)

def test_rotor_state_torque():
    """Verify that apply_torque correctly modifies omega and phase on the SO(2) circle."""
    rotor = RotorState(phase=0.0, omega=1.0, impedance=0.0)
    rotor.apply_torque(1.0, 0.1)
    assert rotor.omega == 1.1
    assert abs(rotor.phase - 0.11) < 1e-5

def test_constant_trajectory():
    """Verify constant orbit evaluate returns expected analytical phase."""
    traj = ConstantTrajectory(initial_phase=0.5, base_omega=2.0)
    phase = traj.evaluate(1.5)
    expected = (0.5 + 2.0 * 1.5) % (2 * math.pi)
    assert abs(phase - expected) < 1e-5

def test_local_tensor_theater_engagement():
    """Verify that local theater activates when phases are close, otherwise stays inactive."""
    theater = LocalTensorTheater(engagement_threshold=0.15)

    # Close phases -> engagement active
    assert theater.check_engagement(0.1, 0.12) is True

    # Far phases -> engagement inactive
    assert theater.check_engagement(0.1, 1.5) is False

def test_tensor_operator_tension_gap():
    """Verify that Tension Gap measures prediction error and adjusts stiffness."""
    operator = TensorOperator(initial_coupling_stiffness=0.8, learning_rate=0.1)
    rotor_a = RotorState(phase=0.0, omega=1.0, impedance=0.1)
    rotor_b = RotorState(phase=0.0, omega=1.0, impedance=0.1)

    # No prediction error scenario
    predicted = (0.1, 0.1)
    observed = (0.1, 0.1)
    gap = operator.synchronize_on_tension_gap(predicted, observed, rotor_a, rotor_b)
    assert gap == 0.0

    # Large prediction error scenario
    predicted = (0.1, 0.5)
    observed = (0.1, 0.1)
    initial_stiffness = operator.stiffness
    gap = operator.synchronize_on_tension_gap(predicted, observed, rotor_a, rotor_b)

    assert gap > 0.0
    # Stiffness should have updated due to the non-zero phase error
    assert operator.stiffness != initial_stiffness

def test_adaptive_tensor_gear_system_integration():
    """Verify the full 3-phase cycle in the integrated system with adaptation."""
    system = AdaptiveTensorGearSystem(engagement_threshold=0.15, learning_rate=0.2)
    # Set initial stiffness to 0.8 so that prediction and actual physical transmission differ,
    # requiring the system to self-tune and adapt the operator.
    system.operator.stiffness = 0.8

    # Start in uncoupled state (phases far apart)
    system.rotor_a.phase = 0.0
    system.rotor_b.phase = 2.0

    res = system.step(t=0.0, dt=0.1)
    assert res["theater_active"] == 0.0
    assert res["tension_gap"] == 0.0

    # Simulate contact (phases close together)
    system.rotor_a.phase = 0.1
    system.rotor_b.phase = 0.11

    initial_stiffness = system.operator.stiffness
    res_active = system.step(t=0.1, dt=0.1)
    assert res_active["theater_active"] == 1.0
    # Impendance and stiffness should adapt to absorb the mismatch
    assert res_active["stiffness"] != initial_stiffness
