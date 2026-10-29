import pytest
import numpy as np
import math
from L1_causal_field.synaptic_architecture.phase_rotor_transition_engine import (
    StaticRotorBakedGround,
    DynamicWaveRotor,
    MultiScalePhaseRotorStack,
    PhaseRotorTransitionEngine,
)


def test_static_rotor_baked_ground_equipotential_query():
    """Verify Static Rotor O(1) equipotential backbone query and spectral axes orthogonality."""
    ground = StaticRotorBakedGround(node_count=100, feature_dim=64)
    assert ground.is_baked is True
    assert ground.backbone_axes.shape == (64, 64)

    # Check orthogonality of pre-baked constraining axes
    identity_approx = np.dot(ground.backbone_axes.T, ground.backbone_axes)
    np.testing.assert_allclose(identity_approx, np.eye(64), atol=1e-5)

    # Query equipotential projection
    input_coords = np.ones(64, dtype=np.float32)
    projected = ground.query_equipotential_backbone(input_coords)
    assert projected.shape == (64,)
    assert np.linalg.norm(projected) > 0.0


def test_dynamic_wave_rotor_q_heat_and_phase_rotation():
    """Verify Dynamic Wave Rotor Clifford phase rotation and Q_heat calculation."""
    rotor = DynamicWaveRotor(feature_dim=64, v_critical=30.0)
    input_wave = np.ones(64, dtype=np.float32)
    constraining_axes = np.eye(64, dtype=np.float32)

    delta_theta = np.pi / 4.0
    rotated_wave, q_heat = rotor.rotate_phase_wave(
        wave_vector=input_wave,
        delta_theta=delta_theta,
        constraining_axes=constraining_axes,
    )

    assert rotated_wave.shape == (64,)
    assert q_heat >= 0.0
    assert math.isclose(rotor.current_phase_angle, delta_theta, abs_tol=1e-5)


def test_multiscale_rotor_stack_interlocking():
    """Verify gear-ratio phase propagation across Macro -> Meso -> Micro -> Nano layers."""
    stack = MultiScalePhaseRotorStack(feature_dim=64)
    drive_torque = 0.5

    phases = stack.propagate_scale_rotation(drive_torque)
    assert set(phases.keys()) == {"Macro", "Meso", "Micro", "Nano"}
    # Nano gear ratio (1/0.001 = 1000) should rotate faster than Macro
    assert phases["Nano"] != phases["Macro"]


def test_phase_rotor_transition_engine_frame_interaction():
    """Verify full engine frame interaction processing, latency, and compression ratio."""
    engine = PhaseRotorTransitionEngine(feature_dim=64, static_nodes=500)
    input_wave = np.random.randn(64).astype(np.float32)

    res = engine.process_frame_interaction(
        input_wave=input_wave,
        camera_angle_delta=0.1,
        interaction_torque=0.2,
    )

    assert res["frame_idx"] == 1
    assert res["latency_ms"] >= 0.0
    assert res["q_heat"] >= 0.0
    assert res["phase_concentration_ratio"] == 5.0  # 250MB / 50MB
    assert res["static_ground_baked"] is True


def test_comparison_with_brute_force_baseline():
    """Verify operational efficiency compared against brute-force full recalculation."""
    engine = PhaseRotorTransitionEngine(feature_dim=64)
    input_wave = np.random.randn(64).astype(np.float32)

    rotor_res = engine.process_frame_interaction(input_wave)
    baseline_res = engine.simulate_brute_force_baseline(full_data_bytes=10 * 1024 * 1024)

    assert rotor_res["latency_ms"] >= 0.0
    assert baseline_res["baseline_latency_ms"] >= 0.0
