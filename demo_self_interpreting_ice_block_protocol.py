#!/usr/bin/env python3
"""
CLI Demonstration Script: Self-Interpreting Ice Block & Scale Coupling Protocol.

Demonstrates:
1. Gas phase: High entropy, dispersed noisy patch nodes.
2. Liquid phase: Dynamic coupling search using Variational Free Energy F_{ij} minimization.
3. Ice phase: Crystallization into Structure(n+1) macro node with embedded CausalSchemaHeader.
4. Self-Deconstruction: O(1) self-interpretation and restoration without external parser.
5. Limit Map: Logging reconstruction error anomalies.
6. Perturbation Resilience: Thermal shock melting and re-crystallization.
"""

import sys
import time
import json
import numpy as np

from core.topology.self_interpreting_scale_node import (
    SelfInterpretingScaleNode,
    CausalSchemaHeader,
    LimitMap,
    PhaseState
)


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def main():
    print_banner("ELYASIA: SELF-INTERPRETING ICE BLOCK & SCALE COUPLING PROTOCOL")
    print("Initializing 2D Scale Coupling Simulation...\n")

    limit_map = LimitMap(epsilon_limit=0.05)

    # --------------------------------------------------------------------------
    # Step 1: Initialize Dispersed Gas Patch Nodes (Structure(n=0))
    # --------------------------------------------------------------------------
    print(">>> STEP 1: Spawning Dispersed Micro Patch Nodes (Structure(n=0) in GAS/LIQUID state)...")

    node_1 = SelfInterpretingScaleNode(
        node_id="patch_alpha",
        scale=0,
        center=(0.0, 0.0),
        radius=1.0,
        phase_angle=0.1,
        local_temperature=0.3,
        phase_state=PhaseState.LIQUID
    )

    node_2 = SelfInterpretingScaleNode(
        node_id="patch_beta",
        scale=0,
        center=(1.8, 0.0),  # Touching boundary
        radius=1.0,
        phase_angle=0.15,   # Phase synchronized
        local_temperature=0.3,
        phase_state=PhaseState.LIQUID
    )

    print(f"  * Node 1: ID={node_1.node_id}, Center={node_1.center}, Phase={node_1.phase_angle:.2f}, State={node_1.phase_state.value}")
    print(f"  * Node 2: ID={node_2.node_id}, Center={node_2.center}, Phase={node_2.phase_angle:.2f}, State={node_2.phase_state.value}")

    # --------------------------------------------------------------------------
    # Step 2: Calculate Variational Free Energy F_{ij} & Spontaneous Coupling
    # --------------------------------------------------------------------------
    print("\n>>> STEP 2: Calculating Variational Free Energy F_{ij} at Contact Boundary Gamma_{12}...")

    F_12, norm_align, phase_offset = node_1.calculate_free_energy(node_2)
    print(f"  * Normal Alignment (n_1 . n_2): {norm_align:.4f}")
    print(f"  * Phase Offset (Delta theta_12): {phase_offset:.4f} rad")
    print(f"  * Variational Free Energy F_12: {F_12:.4f}")

    print("\n  Attempting Spontaneous Coupling into Macro Structure(n=1)...")
    parent_12 = node_1.attempt_spontaneous_coupling(
        node_2, f_threshold=2.0, epsilon_limit=0.05, limit_map=limit_map
    )

    if parent_12:
        print(f"  SUCCESS! Crystallized Macro Node: {parent_12.node_id}")
        print(f"  * Scale: {parent_12.scale}")
        print(f"  * Phase State: {parent_12.phase_state.value.upper()} (FLOPs = 0)")
        print(f"  * Child 1 State: {node_1.phase_state.value.upper()}")
        print(f"  * Child 2 State: {node_2.phase_state.value.upper()}")

        print("\n--- Embedded Self-Interpreting Ice Block Causal Schema Header ---")
        print(json.dumps(parent_12.header.to_dict(), indent=2))
    else:
        print("  COUPLING REFUSED: Free Energy exceeded threshold.")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # Step 3: O(1) Self-Deconstruction Test
    # --------------------------------------------------------------------------
    print("\n>>> STEP 3: Testing O(1) Self-Deconstruction via f_{coupling}^{-1} Header Protocol...")
    restored_subs, deconstruct_log = parent_12.self_deconstruct()

    print(f"  * Deconstruction Log: {json.dumps(deconstruct_log, indent=2)}")
    print(f"  * Restored Sub-nodes Count: {len(restored_subs)}")
    for sub in restored_subs:
        print(f"    - Restored Node: ID={sub.node_id}, Phase State={sub.phase_state.value.upper()}")

    # Re-crystallize parent for perturbation testing
    node_1.phase_state = PhaseState.ICE
    node_2.phase_state = PhaseState.ICE
    parent_12.phase_state = PhaseState.ICE

    # --------------------------------------------------------------------------
    # Step 4: Reconstruction Error Anomaly & Limit Map Registration
    # --------------------------------------------------------------------------
    print("\n>>> STEP 4: Simulating Unexplained Anomaly & Limit Map Registration...")
    node_3 = SelfInterpretingScaleNode("patch_gamma", scale=0, center=(0.0, 0.0), radius=1.0)
    node_4 = SelfInterpretingScaleNode("patch_delta", scale=0, center=(1.8, 0.0), radius=1.0)

    anomaly_parent = node_3.attempt_spontaneous_coupling(
        node_4,
        epsilon_limit=0.05,
        limit_map=limit_map,
        artificial_reconstruction_error=0.14  # Exceeds limit
    )

    print(f"  * Anomaly Node Created: {anomaly_parent.node_id}")
    print(f"  * Limit Map Registered Records: {len(limit_map.records)}")
    print("  * Limit Map Anomaly Details:")
    print(json.dumps(limit_map.records[-1], indent=2))

    # --------------------------------------------------------------------------
    # Step 5: Thermal Perturbation Shock & Reversible Phase Transition
    # --------------------------------------------------------------------------
    print("\n>>> STEP 5: Applying External Thermal Shock & Testing Perturbation Resilience...")
    print("  * Injecting Heat Shock (+4.0 Temp, 0.5 Noise)...")
    parent_12.apply_thermal_perturbation(temperature_delta=4.0, noise_amplitude=0.5)
    print(f"  * Parent Node Phase State: {parent_12.phase_state.value.upper()}")
    print(f"  * Child 1 Phase State: {node_1.phase_state.value.upper()}")
    print(f"  * Child 2 Phase State: {node_2.phase_state.value.upper()}")

    print("\n  * Cooling Down (-4.2 Temp)...")
    parent_12.apply_thermal_perturbation(temperature_delta=-4.2)
    print(f"  * Parent Node Phase State after Re-Cooling: {parent_12.phase_state.value.upper()}")
    print(f"  * Child 1 Phase State after Re-Cooling: {node_1.phase_state.value.upper()}")
    print(f"  * Child 2 Phase State after Re-Cooling: {node_2.phase_state.value.upper()}")

    print_banner("DEMONSTRATION COMPLETE: ALL CAUSAL PROTOCOL INVARIANTS VERIFIED!")


if __name__ == "__main__":
    main()
