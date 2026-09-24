#!/usr/bin/env python3
r"""
[Demo] Meta-Causal Canopy Demonstration (인과적 하늘 엔진 시연)
Demonstrates:
1. Ingestion of raw unanchored packets, files, and function stacks.
2. Teleological Anchor Mapping ("What for" context assignment).
3. Negative Indentation Absorption of phase errors (q_err) without system crashes.
4. Macro-observation of system Order Parameter (\eta) across GAS -> LIQUID -> ICE phase states.
5. Critical Pressure (P_crit) field steering driving mirror symmetry and positive crystallization.
"""

import numpy as np
from core.physics.meta_causal_canopy import (
    MetaCausalCanopy,
    SignalType,
    CausalPhaseState
)


def run_demo():
    print("=" * 80)
    print("      ELYSIA: META-CAUSAL CANOPY (인과적 하늘 엔진) DEMO")
    print("=" * 80)

    canopy = MetaCausalCanopy(dimensions=16, base_critical_pressure=0.5)

    print("\n[Phase 1] Establishing Teleological Anchors ('고정축' - Invariant Purpose)")
    anchor_world = canopy.register_teleological_anchor(
        anchor_id="ANCHOR_WORLD_STATE_SYNC",
        description="MMORPG Causal World State & Entity Position Synchronization",
        intent_vector=np.array([1.0, 0.2, 0.0, 0.0] + [0.0] * 12, dtype=np.float32)
    )
    anchor_save = canopy.register_teleological_anchor(
        anchor_id="ANCHOR_PERSISTENT_STORAGE",
        description="Transactional World File Provenance & DB Integrity",
        intent_vector=np.array([0.0, 1.0, 0.3, 0.0] + [0.0] * 12, dtype=np.float32)
    )
    print(f"  * Registered Anchor 1: {anchor_world.anchor_id} ({anchor_world.description})")
    print(f"  * Registered Anchor 2: {anchor_save.anchor_id} ({anchor_save.description})")

    print("\n[Phase 2] Ingesting & Mapping Raw Bytes, Packets, Files to Teleology ('What for')")
    incoming_data = [
        ("PKT_001", SignalType.PACKET, {"type": "CHAR_POS", "x": 120, "y": 450}, "ANCHOR_WORLD_STATE_SYNC"),
        ("PKT_002", SignalType.PACKET, {"type": "CHAR_ATTACK", "target": "BOSS_01"}, "ANCHOR_WORLD_STATE_SYNC"),
        ("FILE_001", SignalType.FILE, "/var/save/world_chunk_004.bin", "ANCHOR_PERSISTENT_STORAGE"),
        ("STACK_001", SignalType.FUNCTION_STACK, "execute_physics_step(dt=0.016)", "ANCHOR_WORLD_STATE_SYNC"),
        ("NOISE_001", SignalType.RAW_WAVE, "CORRUPTED_HARDWARE_INTERRUPT_0xFE", None),
    ]

    for sig_id, sig_type, payload, target_anchor in incoming_data:
        sig = canopy.map_signal_to_teleology(
            signal_id=sig_id,
            signal_type=sig_type,
            payload=payload,
            target_anchor_id=target_anchor
        )
        print(f"  + Signal [{sig.signal_id:<10}] | Type: {sig.signal_type.value:<15} | Purpose: {sig.teleological_purpose}")
        print(f"    - Initial Phase Error (q_err): {sig.phase_error:.4f}")

    print("\n[Phase 3] Negative Indentation Absorption (Absorbing q_err without System Crash)")
    for sig in canopy.signals:
        if sig.phase_error > 0.3:
            record = canopy.absorb_perturbation_into_manifold(sig)
            print(f"  ~ Absorbed [{sig.signal_id}] into Manifold | Indentation Depth: {record['negative_indentation_depth']:.4f} | Status: {record['status']}")

    report_pre = canopy.get_macro_state_report()
    print(f"\n[Phase 4] Macro-Observation (Macro Phase State Pre-Steering)")
    print(f"  Order Parameter (\u03b7): {report_pre['order_parameter_eta']:.4f}")
    print(f"  System Phase State: {report_pre['phase_state']}")
    print(f"  Total Absorbed q_err: {report_pre['total_absorbed_q_err']:.4f}")

    print("\n[Phase 5] Steering Field Pressure (P_crit Dial Adjustment & Mirror Crystallization)")
    for i in range(1, 4):
        steering_res = canopy.steer_field_pressure(delta_p_crit=1.2)
        print(f"  Step {i}: P_crit = {steering_res['p_crit']:.2f} | Order Parameter (\u03b7) = {steering_res['order_parameter_eta']:.4f} | Phase = {steering_res['phase_state']}")

    report_post = canopy.get_macro_state_report()
    print(f"\n[Phase 6] Final System Macro State")
    print(f"  Order Parameter (\u03b7): {report_post['order_parameter_eta']:.4f}")
    print(f"  Phase State: {report_post['phase_state']}")
    print(f"  Crystallized ICE State Reached: {report_post['is_crystallized_ice']}")

    print("\n" + "=" * 80)
    print("      META-CAUSAL CANOPY DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
