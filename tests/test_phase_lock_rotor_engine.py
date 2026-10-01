"""
Unit tests for Elysia Phase Lock Rotor Engine, 4-Layer Volitional Architecture, and Predictive Resonance.
"""

import numpy as np
from core.consciousness.phase_lock_rotor_engine import (
    ElysiaRotorEngine,
    TensorPhaseLockEngine,
    VolitionalGatedArchitecture,
    PredictiveResonanceGatedEngine
)


def test_elysia_3d_rotor_engine_convergence():
    """Tests 3D vector field phase friction and rotor recalibration towards phase lock."""
    engine = ElysiaRotorEngine(gamma=3.0, beta=0.1)

    np.random.seed(42)
    V_base = np.random.randn(20, 3)
    # Apply a known rotation around Z-axis (45 degrees) to generate external reality
    angle = np.pi / 4.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R_target = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a,  cos_a, 0.0],
        [0.0,    0.0,   1.0]
    ])
    V_out = np.dot(V_base, R_target.T)

    initial_error, _ = engine.step(V_base, V_out, dt=0.001)

    # Run recalibration steps
    errors = []
    for _ in range(50):
        err, tau = engine.step(V_base, V_out, dt=0.05)
        errors.append(err)

    # Verify that error decreases over time (phase lock convergence)
    assert errors[-1] < initial_error, f"Error did not decrease: {initial_error} -> {errors[-1]}"


def test_tensor_phase_lock_engine_so_n():
    """Tests N-dimensional Lie algebra so(N) skew-symmetric torque and SO(N) matrix exponential."""
    N = 8
    engine = TensorPhaseLockEngine(dim_feature=N, gamma=2.5, beta=0.15)

    np.random.seed(123)
    T_base = np.random.randn(10, N)
    T_out = np.random.randn(10, N)

    err1, R_mat1 = engine.step(T_base, T_out, dt=0.01)

    # Check that R_mat is orthogonal: R @ R.T ~= I
    eye_diff = np.linalg.norm(np.dot(R_mat1, R_mat1.T) - np.eye(N))
    assert eye_diff < 1e-4, f"R_mat is not orthogonal: diff={eye_diff}"

    # Check skew-symmetry of omega
    Omega = engine.get_skew_symmetric_omega()
    skew_diff = np.linalg.norm(Omega + Omega.T)
    assert skew_diff < 1e-12, f"Omega is not skew-symmetric: diff={skew_diff}"


def test_volitional_gated_architecture():
    """Tests 4-Layer Volitional Architecture (Environment L0 vs Volitional Agency L2)."""
    arch = VolitionalGatedArchitecture(topology_dim=16, c_max=0.35, tau_min=0.05)

    np.random.seed(55)
    internal_state = np.random.randn(16)

    # 1. Passive match (x_input == internal_state -> prediction_error = 0 < tau_min)
    res_passive = arch.forward(internal_state.copy(), internal_state)
    assert res_passive["passive_gated"]
    assert not res_passive["volitional_active"]
    assert res_passive["compute_cost"] == 0.0

    # 2. Discrepancy triggers Volitional Gate (Layer 2 active, compute bounded by C_max)
    x_input_distorted = internal_state + np.random.randn(16) * 1.5
    res_active = arch.forward(x_input_distorted, internal_state)
    assert not res_active["passive_gated"]
    assert res_active["volitional_active"]
    assert res_active["compute_cost"] <= 0.35 * np.sqrt(16) + 1e-5


def test_predictive_resonance_gated_engine_flow_zones():
    """Tests 4-stage event-driven gating (boredom, flow, panic zones and L_flow calculation)."""
    engine = PredictiveResonanceGatedEngine(
        dim_feature=16,
        tau_min=0.05,
        c_max=0.35
    )

    np.random.seed(99)
    T_base = np.random.randn(5, 16)

    # 1. Low discrepancy -> Boredom zone (alpha = 0, passive resonance)
    res_boredom = engine.process(T_base, T_base.copy(), dt=0.01)
    assert res_boredom["flow_metrics"]["zone"] == "BOREDOM"
    assert res_boredom["compute_gating_alpha"] == 0.0
    assert not res_boredom["dissonance_triggered"]

    # 2. Moderate discrepancy -> Flow zone (dissonance triggered, active compute)
    T_out_flow = T_base + np.random.randn(5, 16) * 0.5
    res_flow = engine.process(T_base, T_out_flow, dt=0.01)
    assert res_flow["flow_metrics"]["zone"] in ["FLOW", "PANIC"]
    assert res_flow["dissonance_triggered"]
    assert res_flow["compute_gating_alpha"] > 0.0
    assert res_flow["internalized"]


if __name__ == "__main__":
    test_elysia_3d_rotor_engine_convergence()
    test_tensor_phase_lock_engine_so_n()
    test_volitional_gated_architecture()
    test_predictive_resonance_gated_engine_flow_zones()
    print("ALL PHASE LOCK ROTOR ENGINE TESTS PASSED!")
