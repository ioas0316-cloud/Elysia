r"""
ELYSIA Planetary Scale-Rotor Distributed Causal Architecture Demo (v2.3+)
==========================================================================

Outputs sequential verification across Scenarios A, B, C plus Mutual Spherical
Mirror jitter absorption, dropped-packet self-healing, signed HMAC governance
patches, Ecosystem Re-Causalization, Parallel Universe projection, and CausalField.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.orchestration.planetary_causal_distributed_engine import (
    PlanetaryEcosystemLoop,
    PlanetaryQuorumNetwork,
    fold_to_core,
    sync_to_causal_field,
    unfold_to_boundary,
)
from core.physics.causal_field import CausalField


def run_demo() -> None:
    print("=" * 94)
    print("    ELYSIA PLANETARY SCALE-ROTOR DISTRIBUTED CAUSAL ENGINE (v2.3) VERIFICATION")
    print("=" * 94)

    net = PlanetaryQuorumNetwork()
    net.broadcast_action("MOVE", 30)
    net.broadcast_action("ACCUMULATE", 25)
    net.broadcast_action("ENHANCE", 1)
    q0 = net.validate_quorum()
    print(f"\n[INIT] 4-Node Quorum Consensus={q0['consensus_reached']} | Hash={q0['consensus_hash']}")

    # ------------------------------------------------------------------
    # SCENARIO A: VirtualRelationalMemory O(1) Axis Switch (0 Recompute)
    # ------------------------------------------------------------------
    node_a = net.nodes["Node_A"]
    node_a.vmem.bind_entity("hero_elysia", node_a.hard_state, node_a.soft_state, node_a.provenance)
    print("\n[SCENARIO A] VIRTUAL RELATIONAL MEMORY PERSPECTIVE AXIS SWITCH (0 RECOMPUTE)")
    print("-" * 94)
    for ax in ("vision", "language", "code", "mechanics"):
        view = node_a.vmem.switch_axis("hero_elysia", ax)
        print(f"  Axis={ax:<9} -> {view}")
    print(
        f"  Axis Switches={node_a.vmem.axis_switch_count} | "
        f"Recomputations={node_a.vmem.recompute_count}"
    )

    # ------------------------------------------------------------------
    # SCENARIO B: FastForward + Anomaly Reject + Cheat Quorum Rollback
    # ------------------------------------------------------------------
    print("\n[SCENARIO B] FLOW-DIRECT FAST-FORWARD, ANOMALY REJECT & BYZANTINE ROLLBACK")
    print("-" * 94)
    net.broadcast_fast_forward(steps=50, action="MOVE", param=15)
    q_ff = net.validate_quorum()
    print(
        f"  FastForward(50 ticks): tick={node_a.hard_state.tick}, orbit={node_a.hard_state.orbit_slot}, "
        f"consensus={q_ff['consensus_reached']}"
    )
    try:
        node_a.execute_fast_forward(steps=10, action="MOVE", param=15, anomaly_check=lambda *_: False)
    except ValueError as exc:
        print(f"  Anomaly Guardrail Triggered: {exc}")

    net.nodes["Node_B"].inject_cheat("enhance_without_accumulate")
    print(f"  Injected Non-Causal Cheat on Node_B -> enhancement={net.nodes['Node_B'].hard_state.enhancement_level}")
    q_cheat = net.validate_quorum()
    print(
        f"  Quorum Rejected: {q_cheat['rejected_nodes']} | "
        f"Node_B Rollbacks={net.nodes['Node_B'].rollback_count}, "
        f"Restored Consensus={q_cheat['consensus_reached']}, enh={net.nodes['Node_B'].hard_state.enhancement_level}"
    )

    # ------------------------------------------------------------------
    # SCENARIO C: Core Fold -> Boundary Unfold (100% Provenance Intact)
    # ------------------------------------------------------------------
    print("\n[SCENARIO C] CORE FOLD (0/1) -> BOUNDARY UNFOLD & 100% PROVENANCE TRACE")
    print("-" * 94)
    core_pkt = fold_to_core(node_a.hard_state, node_a.soft_state, node_a.provenance)
    unfolded = unfold_to_boundary(
        core_pkt, node_a.vmem, "hero_unfolded", node_a.skeleton, node_a.genesis_state
    )
    print(
        f"  Core Phase={core_pkt['core_axis_phase']:.4f} rad | "
        f"Unfolded Provenance Intact={unfolded['provenance_intact']} "
        f"(lineage={unfolded['lineage_length']}, tip={unfolded['tip_hash']})"
    )

    # ------------------------------------------------------------------
    # MUTUAL SPHERICAL MIRROR: Phase Jitter vs Packet Healing & Signed Patch
    # ------------------------------------------------------------------
    print("\n[MIRROR & GOVERNANCE] JITTER ABSORPTION, PACKET HEALING & SIGNED PATCH")
    print("-" * 94)
    net.nodes["Node_C"].inject_phase_jitter(0.18)
    lock_res = net.sync_complex_phase_lock()
    print(f"  Phase-Lock Absorbed Jitter: {lock_res['jitter_absorbed_nodes']} (target={lock_res['target_phase_rad']} rad)")

    rec_peer = net.nodes["Node_B"].execute_action("MOVE", 37)
    net.nodes["Node_C"].execute_action("MOVE", 37)
    net.nodes["Node_D"].execute_action("MOVE", 37)
    heal_res = net.heal_dropped_packet("Node_A", rec_peer)
    print(f"  Self-Healed Dropped Packet on Node_A: {heal_res}")

    secret = "elysia_sovereign_key_2026"
    patch = {"enhance_cost": 15, "max_move_step": 90}
    bad_ok, bad_msg = net.skeleton.apply_patch(patch, "forged_sig", secret)
    sig = net.skeleton.sign_patch(patch, secret)
    good_ok, good_msg = net.skeleton.apply_patch(patch, sig, secret)
    print(f"  Forged Patch={bad_ok} ({bad_msg}) | Signed Patch={good_ok} ({good_msg})")

    # ------------------------------------------------------------------
    # ECOSYSTEM RE-CAUSALIZATION, PARALLEL UNIVERSE & CAUSAL FIELD
    # ------------------------------------------------------------------
    print("\n[ECOSYSTEM & CAUSAL FIELD] MACRO-MICRO RE-CAUSALIZATION & PARALLEL UNIVERSE")
    print("-" * 94)
    eco = PlanetaryEcosystemLoop(skeleton=net.skeleton)
    eco.step_ecosystem(dt_ticks=1, user_free_will_stimulus=("ACCUMULATE", 20))
    s_next = eco.step_ecosystem(dt_ticks=3)
    par = eco.project_parallel_universe(("ACCUMULATE", 25), horizon_ticks=4)
    print(
        f"  Reinterpreted Resonance={s_next['reinterpreted_memories'][0]['reinterpreted_resonance']} | "
        f"Primary Future={s_next['future_orbit_preview']}"
    )
    print(
        f"  Parallel Universe Branch={par['parallel_orbits']} "
        f"(orbital divergence={par['orbital_divergence']})"
    )

    cf = CausalField(dimensions=3)
    sync = sync_to_causal_field(net, ecosystem=eco, causal_field=cf, prefix="planet")
    cf.step(dt=0.05)
    print(f"  CausalField Sync: voxels={len(cf.voxels)}, beams={sync['beams_linked']} -> cf.step OK")
    print("=" * 94)


if __name__ == "__main__":
    run_demo()
