import pytest
import numpy as np
from core.physics.telos_gravitational_adapter import (
    TelosGravitationalBoundaryAdapter,
    TelosSignal,
    ExternalNoiseStream
)


def test_telos_activation_and_deactivation():
    adapter = TelosGravitationalBoundaryAdapter(dimensions=16)
    assert adapter.active_telos is None
    assert adapter.telos_semantic_mass == 0.1

    telos_vec = np.array([1.0] + [0.0] * 15, dtype=np.float32)
    signal = TelosSignal(
        signal_id="TELOS_GAME_LOOP_01",
        intent_description="Primary Game Engine Execution Field",
        telos_vector=telos_vec,
        priority_weight=2.0
    )

    act_res = adapter.activate_telos(signal)
    assert act_res["status"] == "Telos_Activated"
    assert adapter.active_telos is not None
    assert adapter.telos_semantic_mass > 1.0
    assert adapter.field_curvature > 1.0

    deact_res = adapter.deactivate_telos()
    assert deact_res["status"] == "Telos_Deactivated"
    assert adapter.active_telos is None
    assert adapter.telos_semantic_mass == 0.1


def test_background_noise_filtering_without_and_with_telos():
    adapter = TelosGravitationalBoundaryAdapter(dimensions=16)

    noise_vec = np.array([0.0, 1.0] + [0.0] * 14, dtype=np.float32)
    noise_stream = ExternalNoiseStream(
        stream_id="NOISE_OS_TELEMETRY_01",
        source_type="OS_Telemetry",
        signal_vector=noise_vec,
        amplitude=10.0
    )

    # 1. Filter without active Telos (Baseline: no attenuation)
    res_no_telos = adapter.filter_background_noise(noise_stream)
    assert res_no_telos["attenuation_factor"] == 0.0
    assert res_no_telos["retained_amplitude"] == 10.0

    # 2. Activate Telos
    telos_vec = np.array([1.0] + [0.0] * 15, dtype=np.float32)
    signal = TelosSignal(
        signal_id="TELOS_GAME_LOOP_01",
        intent_description="Primary Game Loop Intent",
        telos_vector=telos_vec,
        priority_weight=2.5
    )
    adapter.activate_telos(signal)

    # 3. Filter with active Telos (Attenuates & distorts noise trajectory)
    res_with_telos = adapter.filter_background_noise(noise_stream)
    assert res_with_telos["attenuation_factor"] > 0.5
    assert res_with_telos["retained_amplitude"] < 5.0
    assert res_with_telos["retained_amplitude"] < res_no_telos["retained_amplitude"]


def test_primary_stream_shortcut_routing():
    adapter = TelosGravitationalBoundaryAdapter(dimensions=16)

    payload = {"frame_id": 1001, "render_command": "DRAW_WORLD"}
    res_base = adapter.route_primary_stream(raw_data=payload)
    assert res_base["route_status"] == "ZeroCopy_Shortcut_HighSpeed"
    assert res_base["snr_gain_factor"] == 1.0 + 0.1 * 1.5

    # Activate Telos -> Gain factor amplifies
    telos_vec = np.ones(16, dtype=np.float32)
    signal = TelosSignal(
        signal_id="TELOS_CORE",
        intent_description="Core Intent",
        telos_vector=telos_vec,
        priority_weight=2.0
    )
    adapter.activate_telos(signal)

    res_telos = adapter.route_primary_stream(raw_data=payload)
    assert res_telos["snr_gain_factor"] > res_base["snr_gain_factor"]


def test_system_topology_evaluation():
    adapter = TelosGravitationalBoundaryAdapter(dimensions=16)

    telos_vec = np.array([1.0] + [0.0] * 15, dtype=np.float32)
    signal = TelosSignal(
        signal_id="TELOS_MAIN",
        intent_description="Main Intent",
        telos_vector=telos_vec,
        priority_weight=1.5
    )
    adapter.activate_telos(signal)

    for i in range(5):
        noise = ExternalNoiseStream(
            stream_id=f"NOISE_{i}",
            source_type="Interrupt",
            signal_vector=np.random.randn(16).astype(np.float32),
            amplitude=5.0
        )
        adapter.filter_background_noise(noise)

    topo = adapter.evaluate_system_topology()
    assert topo["telos_active"] is True
    assert topo["total_filtered_streams"] == 5
    assert topo["average_noise_attenuation"] > 0.0
