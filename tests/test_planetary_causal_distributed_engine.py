import pytest

from core.orchestration.planetary_causal_distributed_engine import (
    CausalSkeleton,
    PlanetaryEcosystemLoop,
    PlanetaryQuorumNetwork,
    VirtualRelationalMemory,
    fold_to_core,
    sync_to_causal_field,
    unfold_to_boundary,
)
from core.physics.causal_field import CausalField


def test_scenario_a_virtual_relational_memory_zero_recompute_axis_switch() -> None:
    sk = CausalSkeleton()
    h0 = sk.genesis_state()
    from core.orchestration.planetary_causal_distributed_engine import SoftSpectralState

    s0 = SoftSpectralState()
    h1, s1, r1 = sk.apply_pure_transition(h0, s0, "MOVE", 45, witness_tag="test")
    h2, s2, r2 = sk.apply_pure_transition(h1, s1, "ACCUMULATE", 25, witness_tag="test")

    vmem = VirtualRelationalMemory()
    vmem.bind_entity("avatar_1", h2, s2, [r1, r2])

    vis = vmem.switch_axis("avatar_1", "vision")
    lang = vmem.switch_axis("avatar_1", "language")
    code = vmem.switch_axis("avatar_1", "code")
    mech = vmem.switch_axis("avatar_1", "mechanics")

    assert vmem.axis_switch_count == 4
    assert vmem.recompute_count == 0
    assert vis["orbital_angle_deg"] == 4.5
    assert "ACCUMULATE" in lang["narrative_summary"]
    assert code["tip_hash"] == r2.state_hash
    assert mech["orbit_slot"] == 45 and mech["energy"] == 25


def test_scenario_b_fast_forward_provenance_and_non_causal_cheat_quorum_rollback() -> None:
    net = PlanetaryQuorumNetwork()
    net.broadcast_action("ACCUMULATE", 25)
    net.broadcast_action("ENHANCE", 1)
    q_init = net.validate_quorum()
    assert q_init["consensus_reached"] is True

    # Fast-forward 50 steady ticks across all 4 nodes
    net.broadcast_fast_forward(steps=50, action="MOVE", param=12)
    q_ff = net.validate_quorum()
    assert q_ff["consensus_reached"] is True
    assert net.nodes["Node_A"].hard_state.tick == 52
    assert net.nodes["Node_A"].hard_state.orbit_slot == 600

    # Anomaly branch detector immediately rejects anomalous fast_forward
    with pytest.raises(ValueError, match="NON_CAUSAL_ANOMALY_BRANCH"):
        net.nodes["Node_A"].execute_fast_forward(
            steps=10, action="MOVE", param=12, anomaly_check=lambda *_: False
        )

    # Inject non-causal cheat on Node_B (enhance without accumulate)
    net.nodes["Node_B"].inject_cheat("enhance_without_accumulate")
    q_cheat = net.validate_quorum()
    assert "Node_B" in q_cheat["rejected_nodes"]
    assert "NON_CAUSAL_FRAUD" in q_cheat["rejected_nodes"]["Node_B"]
    # Quorum automatically rolled Node_B back and restored 4/4 consensus!
    assert net.nodes["Node_B"].rollback_count == 1
    assert q_cheat["consensus_reached"] is True
    assert net.nodes["Node_B"].hard_state.enhancement_level == 1


def test_scenario_c_core_fold_boundary_unfold_100pct_provenance() -> None:
    net = PlanetaryQuorumNetwork()
    net.broadcast_action("MOVE", 30)
    net.broadcast_action("ACCUMULATE", 20)
    net.broadcast_action("ENHANCE", 1)
    net.broadcast_fast_forward(steps=15, action="MOVE", param=10)

    node_a = net.nodes["Node_A"]
    core_pkt = fold_to_core(node_a.hard_state, node_a.soft_state, node_a.provenance)
    unfolded = unfold_to_boundary(
        core_pkt, node_a.vmem, "node_a_hero", node_a.skeleton, node_a.genesis_state
    )

    assert unfolded["provenance_intact"] is True
    assert unfolded["lineage_length"] == 4
    assert unfolded["tip_hash"] == node_a.hard_state.state_hash
    assert set(unfolded["axes"].keys()) == {"vision", "language", "code", "mechanics"}


def test_complex_phase_jitter_absorption_vs_non_causal_rejection() -> None:
    net = PlanetaryQuorumNetwork()
    net.broadcast_action("MOVE", 40)
    net.validate_quorum()

    # Soft complex phase jitter within tolerance on Node_C is smoothly absorbed
    net.nodes["Node_C"].inject_phase_jitter(0.18)
    sync_res = net.sync_complex_phase_lock()
    assert "Node_C" in sync_res["jitter_absorbed_nodes"]
    assert net.nodes["Node_C"].rollback_count == 0
    q_ok = net.validate_quorum()
    assert q_ok["consensus_reached"] is True

    # Non-causal integer tamper on Node_B cannot be smoothed over -> rolled back
    net.nodes["Node_B"].inject_cheat("energy_overflow")
    q_bad = net.validate_quorum()
    assert "Node_B" in q_bad["rejected_nodes"]
    assert net.nodes["Node_B"].rollback_count == 1


def test_mutual_spherical_mirror_packet_loss_self_healing_vs_fraud() -> None:
    net = PlanetaryQuorumNetwork()
    net.broadcast_action("ACCUMULATE", 22)
    net.validate_quorum()

    # Simulate Node_A missing a legitimate transition packet that Node_B, C, D received
    rec_b = net.nodes["Node_B"].execute_action("MOVE", 37)
    net.nodes["Node_C"].execute_action("MOVE", 37)
    net.nodes["Node_D"].execute_action("MOVE", 37)

    heal_res = net.heal_dropped_packet("Node_A", rec_b)
    assert heal_res["healed"] is True
    assert heal_res["fraud_detected"] is False
    assert heal_res["recovered_action"] == "MOVE"
    assert heal_res["recovered_param"] == 37
    assert net.validate_quorum()["consensus_reached"] is True

    # Impossible cheat disguised as a dropped packet (orbit jump 500 in 1 tick > max_move=60)
    bad_rec = type(rec_b)(
        tick=rec_b.tick + 1,
        prev_hash=rec_b.state_hash,
        action="MOVE",
        param=500,
        steps=1,
        rules_version=1,
        orbit_slot=537,
        energy=22,
        enhancement_level=0,
        state_hash="deadbeef" * 4,
    )
    cheat_heal = net.heal_dropped_packet("Node_A", bad_rec)
    assert cheat_heal["healed"] is False
    assert cheat_heal["fraud_detected"] is True
    assert net.nodes["Node_A"].rollback_count == 1


def test_signed_hmac_patch_authority_and_tamper_rejection() -> None:
    sk = CausalSkeleton()
    secret = "elysia_sovereign_key_2026"
    patch = {"enhance_cost": 15, "max_move_step": 90}

    ok_bad, msg_bad = sk.apply_patch(patch, "forged_bad_signature_hex", secret)
    assert ok_bad is False
    assert sk.rules_version == 1
    assert sk.enhance_cost == 20

    sig = sk.sign_patch(patch, secret)
    ok_good, msg_good = sk.apply_patch(patch, sig, secret)
    assert ok_good is True
    assert "v2" in msg_good
    assert sk.rules_version == 2
    assert sk.enhance_cost == 15
    assert sk.max_move_step == 90


def test_ecosystem_recausalization_parallel_universe_and_causal_field_sync() -> None:
    eco = PlanetaryEcosystemLoop()
    s1 = eco.step_ecosystem(dt_ticks=1, user_free_will_stimulus=("ACCUMULATE", 20))
    assert s1["npc_energy"] == 20
    assert len(s1["future_orbit_preview"]) == 4

    # Non-causal user free-will stimulus outside cone must be rejected!
    with pytest.raises(ValueError, match="NON_CAUSAL_TRANSITION"):
        eco.step_ecosystem(dt_ticks=1, user_free_will_stimulus=("MOVE", 999))

    # Advance macro environment -> past interaction re-causalizes smoothly
    s2 = eco.step_ecosystem(dt_ticks=3, user_free_will_stimulus=None)
    assert s2["reinterpreted_memories"][0]["reinterpreted_resonance"] < 1.0

    # Parallel universe simulation does not mutate primary NPC state
    energy_before = eco.npc_hard.energy
    orbit_before = eco.npc_hard.orbit_slot
    par = eco.project_parallel_universe(("ACCUMULATE", 25), horizon_ticks=4)
    assert len(par["parallel_orbits"]) == 4
    assert eco.npc_hard.energy == energy_before
    assert eco.npc_hard.orbit_slot == orbit_before

    # CausalField synchronization passes cf.step cleanly
    net = PlanetaryQuorumNetwork()
    net.broadcast_action("MOVE", 25)
    cf = CausalField(dimensions=3)
    sync = sync_to_causal_field(net, ecosystem=eco, causal_field=cf, prefix="planet_test")
    assert len(cf.voxels) == 5
    assert sync["beams_linked"] == 7
    cf.step(dt=0.05)
