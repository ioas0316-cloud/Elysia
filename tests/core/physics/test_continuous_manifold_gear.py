import pytest
import math
from core.physics.continuous_manifold_gear import (
    ContinuousRotorState,
    ContinuousManifoldGearSystem
)

def test_continuous_rotor_state_torque():
    """Verify continuous rotor torque application scales correctly under impedance."""
    rotor = ContinuousRotorState(phase=0.0, omega=1.0, impedance=0.0)
    rotor.apply_torque(1.0, 0.1)
    assert rotor.omega == 1.1
    assert abs(rotor.phase - 0.11) < 1e-5

    # With non-zero impedance, effective torque is smaller
    rotor_impeded = ContinuousRotorState(phase=0.0, omega=1.0, impedance=1.0)
    rotor_impeded.apply_torque(1.0, 0.1)
    # effective_torque = 1.0 / (1.0 + 1.0) = 0.5
    # omega = 1.0 + 0.5 * 0.1 = 1.05
    assert abs(rotor_impeded.omega - 1.05) < 1e-5

def test_continuous_manifold_gear_transitions():
    """Verify that the coupling weight scales smoothly with metric distance without any if-else branching."""
    system = ContinuousManifoldGearSystem(
        base_omega_a=2.0,
        base_omega_b=2.0,
        sensitivity_radius=0.25
    )

    # 1. Uncoupled state: Phases far apart (diff = pi)
    system.rotor_a.phase = 0.0
    system.rotor_b.phase = math.pi
    res_uncoupled = system.step(t=0.0, dt=0.1)
    # Metric distance is sin^2(pi/2) = 1.0
    # Coupling weight = exp(-1.0 / (2 * 0.25^2)) = exp(-8) approx 0.00033
    assert res_uncoupled["coupling_weight"] < 0.01
    assert res_uncoupled["tension_gap"] < 0.01

    # 2. Coupled state: Phases identical (diff = 0)
    # We set omega to 0 so the phase doesn't drift during the step
    system.rotor_a.omega = 0.0
    system.rotor_b.omega = 0.0
    system.rotor_a.phase = 0.1
    system.rotor_b.phase = 0.1
    res_coupled = system.step(t=0.1, dt=0.1)
    # Metric distance is 0.0, coupling weight is exp(0) = 1.0
    assert res_coupled["coupling_weight"] == 1.0

def test_continuous_manifold_gear_self_tuning():
    """Verify stiffness and impedance adapt smoothly based on tension gap."""
    system = ContinuousManifoldGearSystem(
        base_omega_a=2.0,
        base_omega_b=2.0,
        sensitivity_radius=0.25,
        learning_rate=0.2
    )
    # Set mismatched initial stiffness
    system.stiffness = 0.5

    # Force a coupled state with phase mismatch to trigger adaptation
    system.rotor_a.phase = 0.1
    system.rotor_b.phase = 0.15

    initial_stiffness = system.stiffness
    res = system.step(t=0.0, dt=0.1)

    # Coupling weight must be non-zero since the phases are close
    assert res["coupling_weight"] > 0.5
    # Tension gap should be non-zero due to stiffness mismatch
    assert res["tension_gap"] > 0.0
    # Stiffness and rotor impedance should have shifted away from initial values
    assert system.stiffness != initial_stiffness
    assert res["rotor_a_impedance"] > 0.01
