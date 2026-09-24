import pytest
import numpy as np
from core.physics.meta_causal_canopy import (
    MetaCausalCanopy,
    SignalType,
    CausalPhaseState
)


def test_meta_causal_canopy_initialization():
    canopy = MetaCausalCanopy(dimensions=16, base_critical_pressure=1.0)
    report = canopy.get_macro_state_report()

    assert report["dimensions"] == 16
    assert report["p_crit"] == 1.0
    assert report["registered_anchors_count"] == 0
    assert report["processed_signals_count"] == 0


def test_anchor_registration_and_teleological_mapping():
    canopy = MetaCausalCanopy(dimensions=16)

    # 1. Register Teleological Anchor
    intent_vec = np.array([1.0, 0.0, 0.5, 0.0] + [0.0] * 12, dtype=np.float32)
    anchor = canopy.register_teleological_anchor(
        anchor_id="ANCHOR_GAME_WORLD",
        description="Character Position & Causal World State Sync",
        intent_vector=intent_vec
    )

    assert anchor.anchor_id == "ANCHOR_GAME_WORLD"
    assert len(anchor.anchor_vector) == 16

    # 2. Map raw packet signal to teleology
    packet_payload = {"packet_id": 101, "cmd": "MOVE_FORWARD", "x": 10.5, "y": 20.2}
    signal = canopy.map_signal_to_teleology(
        signal_id="PKT_MOVE_101",
        signal_type=SignalType.PACKET,
        payload=packet_payload,
        target_anchor_id="ANCHOR_GAME_WORLD",
        raw_vector=np.array([0.9, 0.1, 0.4, 0.1] + [0.0] * 12, dtype=np.float32)
    )

    assert signal.signal_id == "PKT_MOVE_101"
    assert "ANCHOR_GAME_WORLD" in signal.teleological_purpose
    assert signal.phase_error > 0.0


def test_perturbation_absorption_without_crash():
    canopy = MetaCausalCanopy(dimensions=16)

    # Map noisy/unanchored signal
    noisy_signal = canopy.map_signal_to_teleology(
        signal_id="NOISY_PACKET_999",
        signal_type=SignalType.PACKET,
        payload="CORRUPTED_RAW_BYTES_0xDEADBEEF",
        raw_vector=np.array([-0.8, 0.9, -0.5, 0.7] + [0.2] * 12, dtype=np.float32)
    )

    # Absorb perturbation
    record = canopy.absorb_perturbation_into_manifold(noisy_signal)

    assert record["status"] == "Absorbed_Without_Crash"
    assert record["negative_indentation_depth"] > 0.0
    assert canopy.get_macro_state_report()["absorbed_indentations_count"] == 1


def test_field_steering_and_ice_crystallization():
    canopy = MetaCausalCanopy(dimensions=16, base_critical_pressure=0.5)

    anchor_vec = np.array([1.0, 0.0, 0.0, 0.0] + [0.0] * 12, dtype=np.float32)
    canopy.register_teleological_anchor("ANCHOR_CORE", "Core Execution Anchor", anchor_vec)

    # Add several signals with initial phase error
    for i in range(5):
        raw_vec = np.array([0.5, 0.5, 0.1, 0.1] + [0.1 * i] * 12, dtype=np.float32)
        canopy.map_signal_to_teleology(
            signal_id=f"SIG_{i}",
            signal_type=SignalType.FUNCTION_STACK,
            payload=f"stack_frame_{i}",
            target_anchor_id="ANCHOR_CORE",
            raw_vector=raw_vec
        )

    initial_report = canopy.get_macro_state_report()
    initial_eta = initial_report["order_parameter_eta"]

    # Increase critical pressure P_crit across iterations
    for _ in range(5):
        canopy.steer_field_pressure(delta_p_crit=1.5)

    final_report = canopy.get_macro_state_report()

    # Verify order parameter increased and phase crystallized into ICE
    assert final_report["order_parameter_eta"] > initial_eta
    assert final_report["phase_state"] == CausalPhaseState.ICE.value
    assert final_report["is_crystallized_ice"] is True
