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
    stack = MultiScalePhaseRotorStack(feature_dim=64, data_scale_gb=0.3)
    drive_torque = 0.5

    phases = stack.propagate_scale_rotation(drive_torque)
    assert set(phases.keys()) == {"Macro", "Meso", "Micro", "Nano"}
    # Nano gear ratio (1/0.001 = 1000) should rotate faster than Macro
    assert phases["Nano"] != phases["Macro"]


def test_multigigabyte_scale_layer_expansion():
    """Verify layer depth expansion across 0.3GB, 10GB, 30GB, and 100GB scale configurations."""
    stack_03 = MultiScalePhaseRotorStack(data_scale_gb=0.3)
    stack_10 = MultiScalePhaseRotorStack(data_scale_gb=10.0)
    stack_30 = MultiScalePhaseRotorStack(data_scale_gb=30.0)
    stack_100 = MultiScalePhaseRotorStack(data_scale_gb=100.0)

    assert len(stack_03.layers) == 4
    assert len(stack_10.layers) == 6
    assert len(stack_30.layers) == 8
    assert len(stack_100.layers) == 10


def test_memory_tiering_breakdown_gtx1060():
    """Verify memory tiering calculation under GTX 1060 (3GB VRAM, 16GB RAM) hardware limits."""
    engine_10gb = PhaseRotorTransitionEngine(data_scale_gb=10.0)
    tiering_10gb = engine_10gb.get_memory_tiering_breakdown(vram_limit_gb=3.0, ram_limit_gb=16.0)

    assert tiering_10gb["vram_gb"] == 2.5
    assert tiering_10gb["ram_gb"] == 7.5
    assert tiering_10gb["ssd_cache_gb"] == 0.0  # Fits inside VRAM + RAM

    engine_100gb = PhaseRotorTransitionEngine(data_scale_gb=100.0)
    tiering_100gb = engine_100gb.get_memory_tiering_breakdown(vram_limit_gb=3.0, ram_limit_gb=16.0)

    assert tiering_100gb["vram_gb"] == 2.5
    assert tiering_100gb["ram_gb"] == 13.5
    assert tiering_100gb["ssd_cache_gb"] == 84.0  # Tiered to SSD NVMe cache


def test_phase_rotor_transition_engine_frame_interaction():
    """Verify full engine frame interaction processing, latency, and compression ratio."""
    engine = PhaseRotorTransitionEngine(feature_dim=64, static_nodes=500, data_scale_gb=0.3)
    input_wave = np.random.randn(64).astype(np.float32)

    res = engine.process_frame_interaction(
        input_wave=input_wave,
        camera_angle_delta=0.1,
        interaction_torque=0.2,
    )

    assert res["frame_idx"] == 1
    assert res["latency_ms"] >= 0.0
    assert res["q_heat"] >= 0.0
    assert abs(res["phase_concentration_ratio"] - 6.144) < 1e-3  # (0.3 * 1024) / 50.0 = 6.144
    assert res["static_ground_baked"] is True


def test_comparison_with_brute_force_baseline():
    """Verify operational efficiency compared against brute-force full recalculation."""
    engine = PhaseRotorTransitionEngine(feature_dim=64, data_scale_gb=10.0)
    input_wave = np.random.randn(64).astype(np.float32)

    rotor_res = engine.process_frame_interaction(input_wave)
    baseline_res = engine.simulate_brute_force_baseline(data_scale_gb=10.0)

    assert rotor_res["latency_ms"] >= 0.0
    assert baseline_res["baseline_latency_ms"] > rotor_res["latency_ms"]
