"""
Unit tests for Self-Interpreting Scale Node & Causal Schema Protocol.

Verifies T1 through T5 acceptance criteria:
- T1: Spatial-temporal phase locking and boundary alignment ($F_{ij} < F_{threshold}$).
- T2: Thermal phase transitions controlled by local temperature $T_{local}$.
- T3: Self-deconstruction using embedded $f_{coupling}^{-1}$ schema header without external parser.
- T4: Reconstruction error threshold check and Limit Map registration.
- T5: Perturbation resilience (partial melting and re-crystallization under noise).
"""

import math
import numpy as np
import pytest

from core.topology.self_interpreting_scale_node import (
    SelfInterpretingScaleNode,
    CausalSchemaHeader,
    LimitMap,
    PhaseState
)


def test_t1_spontaneous_coupling_phase_locking():
    """
    T1: Verify spatial boundary alignment and temporal phase locking leading to spontaneous coupling.
    """
    node_a = SelfInterpretingScaleNode(
        node_id="node_a",
        scale=0,
        center=(0.0, 0.0),
        radius=1.0,
        phase_angle=0.1,
        local_temperature=0.2,
        phase_state=PhaseState.LIQUID
    )
    node_b = SelfInterpretingScaleNode(
        node_id="node_b",
        scale=0,
        center=(1.8, 0.0),  # Touching boundaries
        radius=1.0,
        phase_angle=0.15,   # Close phase angle (synchronized)
        local_temperature=0.2,
        phase_state=PhaseState.LIQUID
    )

    # Free energy calculation
    f_ij, norm_align, phase_offset = node_a.calculate_free_energy(node_b)
    assert f_ij < 2.0, f"Free energy {f_ij} should be below threshold 2.0 for synchronized touching nodes"

    # Spontaneous coupling
    parent = node_a.attempt_spontaneous_coupling(node_b, f_threshold=2.0)
    assert parent is not None, "Parent node Structure(n+1) should be formed"
    assert parent.scale == 1, "Scale level should increment from 0 to 1"
    assert parent.phase_state == PhaseState.ICE, "Parent node should be in ICE state"
    assert node_a.phase_state == PhaseState.ICE, "Child A should be crystallized in ICE state"
    assert node_b.phase_state == PhaseState.ICE, "Child B should be crystallized in ICE state"
    assert parent.header is not None, "Embedded Ice Block Causal Schema Header must exist"


def test_t2_thermal_phase_transitions():
    """
    T2: Verify local temperature T_local control over phase transitions (Gas <-> Liquid <-> Ice).
    """
    node_a = SelfInterpretingScaleNode(
        node_id="hot_a",
        scale=0,
        center=(0.0, 0.0),
        radius=1.0,
        phase_angle=0.0,
        local_temperature=4.0,  # High temperature -> agitation
        phase_state=PhaseState.LIQUID
    )
    node_b = SelfInterpretingScaleNode(
        node_id="hot_b",
        scale=0,
        center=(1.8, 0.0),
        radius=1.0,
        phase_angle=3.14,  # Out-of-phase
        local_temperature=4.0,
        phase_state=PhaseState.LIQUID
    )

    # Attempt coupling under high thermal noise -> should fail
    parent = node_a.attempt_spontaneous_coupling(node_b, f_threshold=1.5)
    assert parent is None, "Coupling should be refused under high thermal agitation"
    assert node_a.phase_state == PhaseState.GAS, "Hot node A should disperse into GAS"
    assert node_b.phase_state == PhaseState.GAS, "Hot node B should disperse into GAS"

    # Cool down nodes
    node_a.apply_thermal_perturbation(temperature_delta=-3.8)
    node_b.apply_thermal_perturbation(temperature_delta=-3.8)
    node_a.phase_angle = 0.0
    node_b.phase_angle = 0.05
    node_a.phase_state = PhaseState.LIQUID
    node_b.phase_state = PhaseState.LIQUID

    parent_cool = node_a.attempt_spontaneous_coupling(node_b, f_threshold=2.0)
    assert parent_cool is not None, "Cooling down should enable spontaneous crystallization into ICE"


def test_t3_self_deconstruction_inverse_protocol():
    """
    T3: Verify O(1) self-deconstruction using the embedded f_{coupling}^{-1} schema header without external parser.
    """
    node_a = SelfInterpretingScaleNode("node_1", scale=0, center=(0.0, 0.0), radius=1.0, phase_angle=0.0)
    node_b = SelfInterpretingScaleNode("node_2", scale=0, center=(1.5, 0.0), radius=1.0, phase_angle=0.0)

    parent = node_a.attempt_spontaneous_coupling(node_b)
    assert parent is not None

    # Verify header contains inverse protocol instructions
    inv_proto = parent.header.inverse_protocol()
    assert inv_proto["action"] == "deconstruct"
    assert inv_proto["scale_from"] == 1
    assert inv_proto["scale_to"] == 0

    # Self-deconstruct
    restored_subs, log = parent.self_deconstruct()
    assert len(restored_subs) == 2
    assert restored_subs[0].node_id == "node_1"
    assert restored_subs[1].node_id == "node_2"
    assert restored_subs[0].phase_state == PhaseState.LIQUID
    assert restored_subs[1].phase_state == PhaseState.LIQUID
    assert log["status"] == "success"


def test_t4_limit_map_reconstruction_anomaly():
    """
    T4: Verify Limit Map logging when reconstruction error exceeds epsilon_limit.
    """
    limit_map = LimitMap(epsilon_limit=0.05)

    node_a = SelfInterpretingScaleNode("ano_a", scale=0, center=(0.0, 0.0), radius=1.0, phase_angle=0.0)
    node_b = SelfInterpretingScaleNode("ano_b", scale=0, center=(1.8, 0.0), radius=1.0, phase_angle=0.0)

    # Force high artificial reconstruction error (unexplained variant phenomenon)
    parent = node_a.attempt_spontaneous_coupling(
        node_b,
        epsilon_limit=0.05,
        limit_map=limit_map,
        artificial_reconstruction_error=0.12  # Exceeds 0.05 threshold
    )

    assert parent is not None
    assert len(limit_map.records) == 1, "Anomaly should be registered in Limit Map"
    record = limit_map.records[0]
    assert record["delta_recon"] == 0.12
    assert record["epsilon_limit"] == 0.05
    assert record["node_id"] == parent.node_id


def test_t5_perturbation_resilience():
    """
    T5: Verify perturbation resilience (partial melting and re-crystallization under noise).
    """
    node_a = SelfInterpretingScaleNode("res_a", scale=0, center=(0.0, 0.0), radius=1.0, phase_angle=0.0)
    node_b = SelfInterpretingScaleNode("res_b", scale=0, center=(1.5, 0.0), radius=1.0, phase_angle=0.0)

    parent = node_a.attempt_spontaneous_coupling(node_b)
    assert parent.phase_state == PhaseState.ICE

    # Apply thermal shock / perturbation to parent ice block
    parent.apply_thermal_perturbation(temperature_delta=4.0, noise_amplitude=0.5)
    assert parent.phase_state == PhaseState.LIQUID, "Thermal shock should melt ICE into LIQUID"
    assert node_a.phase_state == PhaseState.LIQUID
    assert node_b.phase_state == PhaseState.LIQUID

    # Cool back down
    parent.apply_thermal_perturbation(temperature_delta=-4.2)
    assert parent.phase_state == PhaseState.ICE, "Re-cooling should re-crystallize parent node back to ICE"
    assert node_a.phase_state == PhaseState.ICE, "Child A should re-crystallize back to ICE"
    assert node_b.phase_state == PhaseState.ICE, "Child B should re-crystallize back to ICE"
