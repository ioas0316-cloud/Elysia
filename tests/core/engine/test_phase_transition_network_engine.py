"""
Unit tests for PhaseTransitionNetworkEngine and 3-Boundary Phase Sync Architecture.
"""

import pytest
import numpy as np
from core.engine.phase_transition_network_engine import (
    PhaseTransitionNetworkEngine,
    PhaseState,
    InvariantAnchor,
    MiddleBoundaryBuffer
)


def test_sub_critical_impact_elastic_vibration():
    """
    약한 충격 (q_err <= V_well):
    상전이 없이 ICE 상태를 유지하며 탄성 울림(Elastic Vibration)으로 미세 완충하는지 검증.
    """
    engine = PhaseTransitionNetworkEngine(entity_id="hero_unit")
    anchor = engine.anchor
    anchor.potential_well_depth = 1.0  # V_well = 1.0

    # P_env = 0.2, delay = 10ms, loss = 0.0 -> q_err ~ 0.22 <= 1.0
    res = engine.simulate_network_frame(
        dt=0.016,
        env_pressure=0.2,
        rtt_ms=10.0,
        packet_loss_rate=0.0,
        packet_arrived=True
    )

    assert res["phase_state"] == PhaseState.ICE.value
    assert res["phase_error"] <= anchor.potential_well_depth


def test_super_critical_liquid_transition_and_fluid_interpolation():
    """
    강한 충격 (지연/지터 발생):
    q_err > V_well 일 때 LIQUID 상전이가 일어나고 멈춤(Freeze) 없는 유동적 궤적 연결을 수행하는지 검증.
    """
    engine = PhaseTransitionNetworkEngine(entity_id="hero_unit")
    engine.update_server_state(intent=(10.0, 0.0, 0.0), base_pos=(0.0, 0.0, 0.0))
    engine.anchor.potential_well_depth = 0.3

    # High delay: rtt = 150ms -> q_err > 0.3
    res = engine.simulate_network_frame(
        dt=0.05,
        env_pressure=0.8,
        rtt_ms=150.0,
        packet_loss_rate=0.05,
        packet_arrived=False
    )

    assert res["phase_state"] == PhaseState.LIQUID.value
    # Positional trajectory should advance smoothly along intent without freeze
    pos = res["position"]
    assert pos[0] > 0.0


def test_super_critical_gas_transition_and_mirror_re_ice():
    """
    극심한 단절/손실 (GAS 상전이) 후 거울 대칭 Re-ICE 복원 검증.
    """
    engine = PhaseTransitionNetworkEngine(entity_id="hero_unit")
    engine.anchor.potential_well_depth = 0.5

    # Severe packet loss & delay -> GAS
    res_gas = engine.simulate_network_frame(
        dt=0.05,
        env_pressure=2.0,
        rtt_ms=350.0,
        packet_loss_rate=0.6,
        packet_arrived=False
    )

    assert res_gas["phase_state"] == PhaseState.GAS.value

    # Packet arrives -> Mirror Symmetry Re-ICE
    engine.middle_boundary.phase_error = 0.4  # Within recoverable bound
    res_re_ice = engine.simulate_network_frame(
        dt=0.016,
        env_pressure=0.1,
        rtt_ms=20.0,
        packet_loss_rate=0.0,
        packet_arrived=True
    )

    assert res_re_ice["re_ice_performed"] is True
    assert res_re_ice["phase_state"] == PhaseState.ICE.value
    assert res_re_ice["phase_error"] == 0.0


def test_bandwidth_reduction():
    """
    고정축-변수축 상전이 구조의 통신량 절감 효과 검증.
    """
    engine = PhaseTransitionNetworkEngine(entity_id="hero_unit")

    for _ in range(100):
        engine.simulate_network_frame(
            dt=0.016,
            env_pressure=0.1,
            rtt_ms=20.0,
            packet_loss_rate=0.0,
            packet_arrived=True
        )

    # Reduction ratio should be ~80% (12 bytes vs 60 bytes)
    ratio = engine.simulate_network_frame(
        dt=0.016,
        env_pressure=0.1,
        rtt_ms=20.0,
        packet_loss_rate=0.0,
        packet_arrived=True
    )["bandwidth_reduction_ratio"]

    assert ratio >= 0.75
