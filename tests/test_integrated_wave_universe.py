"""
Unit Tests for Integrated Wave Universe Core Runtime
=====================================================
Tests spatial Clifford rotor rotations, Kuramoto NPC phase synchronization,
and zero-cost execution vs. dynamic phase re-locking dynamics.
"""

import pytest
import numpy as np
from elysia_core.integrated_wave_universe import (
    SpatialManifold,
    NPCPhaseNode,
    IntegratedWaveUniverse,
)


def test_spatial_manifold_rotor_rotation():
    manifold = SpatialManifold(zone_name="test_meadow", rotor_angle=0.0)
    initial_vector = np.copy(manifold.manifold_vector)

    delta = np.pi / 4.0  # 45 degrees
    new_angle = manifold.rotate_rotor(delta)

    assert pytest.approx(new_angle) == np.pi / 4.0
    # Check that high-dimensional vector projection updated
    assert not np.allclose(manifold.manifold_vector[:2], initial_vector[:2])


def test_npc_kuramoto_phase_locking():
    npc = NPCPhaseNode(
        name="test_npc",
        title="Test NPC",
        natural_frequency=1.0,
        phase=0.0,
    )

    player_freq = 1.0
    player_phase = 0.0

    # Simulate 30 time steps where player phase advances with frequency
    for _ in range(30):
        player_phase = (player_phase + player_freq * 0.05) % (2 * np.pi)
        new_phase, diff = npc.update_kuramoto_phase(
            player_phase=player_phase,
            coupling_k=2.0,
            dt=0.05
        )

    # NPC phase should converge toward player phase under strong coupling
    assert diff < 0.25
    assert len(npc.resonance_history) == 30


def test_integrated_wave_universe_stepping():
    universe = IntegratedWaveUniverse(primary_zone="fantasy_meadow")

    # Step 1: Idle
    t1 = universe.step_universe(player_action="idle", player_speech="Hello", dt=0.05)
    assert t1["spatial_zone"] == "fantasy_meadow"
    assert t1["rotor_angle_rad"] == 0.0
    assert "elysia" in t1["npc_reports"]

    # Step 2: Spatial movement
    t2 = universe.step_universe(player_action="move_forward", player_speech="Moving", dt=0.05)
    assert t2["rotor_angle_rad"] > 0.0
    assert "ROTOR_SPIN" in t2["spatial_status"]

    # Step 3: Zone switch
    t3 = universe.step_universe(player_action="switch_zone_dungeon", dt=0.05)
    assert t3["spatial_zone"] == "dungeon_depths"
    assert "TOPOLOGY_JUMP" in t3["spatial_status"]


def test_kuramoto_order_parameter_convergence():
    universe = IntegratedWaveUniverse(primary_zone="fantasy_meadow")

    # Run for 20 steps with stable input
    for _ in range(20):
        telemetry = universe.step_universe(
            player_action="idle",
            player_speech="",
            player_voice_freq=1.2,
            dt=0.1
        )

    # Swarm order parameter R should remain high (>0.8) as NPC phase locks to player
    assert telemetry["kuramoto_order_R"] > 0.8
