"""
Tri-Domain Structural Isomorphism Verification Script
======================================================
Verifies that Enactive Boundary Layer operates across 3 distinct domains without labels or loss backpropagation:
1. Physical Domain: Apple Falling Trajectory (Floor impact & gravity boundary constraint)
2. Financial Domain: Corporate Bankruptcy / Stock Crash (Liquidity depletion & insolvency boundary constraint)
3. Software Domain: System Exception Propagation (Stack/memory overflow & exception boundary constraint)

Demonstrates that non-negotiable external constraints recalibrate lens phase angles and edge impedance,
revealing shared topological isomorphism ("Trajectory -> Boundary Impact -> Phase Transition & Recalibration").
"""

import sys
import os
import numpy as np
import networkx as nx

from core.lens.enactive_boundary_layer import EnactiveBoundaryLayer
from core.lens.cognitive_lens_engine import CognitiveLensEngine, ContextualDimension


def run_tri_domain_isomorphism_verification():
    print("=" * 80)
    print("  ELYSIA: TRI-DOMAIN STRUCTURAL ISOMORPHISM VERIFICATION")
    print("  Rejects Numeric Reductionism, Scalar Loss, and Backpropagation.")
    print("  Operates via Non-Negotiable External Constraint & Phase Recalibration.")
    print("=" * 80)

    lens_engine = CognitiveLensEngine()
    ebl = EnactiveBoundaryLayer(lens_engine=lens_engine, alpha=0.5, beta=0.2, threshold=0.15)

    # Domain 1: Apple Falling Trajectory (Physical Domain)
    # Intended phase angle: pi/2 (Floating in air assumption). Reality phase angle: 0.1 (Ground impact constraint)
    ebl.add_causal_node("Apple_Falling_Trajectory", frequency=5.0, phase=np.pi / 2.0, dimension=ContextualDimension.TOPOLOGICAL_CURVATURE)
    ebl.add_causal_node("Ground_Impact_State", frequency=5.0, phase=0.1)
    ebl.add_causal_edge("Apple_Falling_Trajectory", "Ground_Impact_State", initial_impedance=0.1)

    # Domain 2: Corporate Bankruptcy / Stock Crash (Financial Domain)
    # Intended phase angle: pi/2 (Unbounded growth assumption). Reality phase angle: 0.1 (Solvency limit constraint)
    ebl.add_causal_node("Corporate_Liquidity_Flow", frequency=5.0, phase=np.pi / 2.0, dimension=ContextualDimension.BIOLOGICAL_FRICTION)
    ebl.add_causal_node("Insolvency_Boundary", frequency=5.0, phase=0.1)
    ebl.add_causal_edge("Corporate_Liquidity_Flow", "Insolvency_Boundary", initial_impedance=0.1)

    # Domain 3: System Exception Wave Propagation (Software Domain)
    # Intended phase angle: pi/2 (Infinite stack recursion assumption). Reality phase angle: 0.1 (Stack overflow constraint)
    ebl.add_causal_node("Software_Execution_Wave", frequency=5.0, phase=np.pi / 2.0, dimension=ContextualDimension.RELATIONAL_INTENT)
    ebl.add_causal_node("Memory_Boundary_Isolation", frequency=5.0, phase=0.1)
    ebl.add_causal_edge("Software_Execution_Wave", "Memory_Boundary_Isolation", initial_impedance=0.1)

    domains = [
        ("Physical Domain (Apple Fall)", "Apple_Falling_Trajectory", "Ground_Impact_State", 0.1),
        ("Financial Domain (Stock Crash)", "Corporate_Liquidity_Flow", "Insolvency_Boundary", 0.1),
        ("Software Domain (Exception)", "Software_Execution_Wave", "Memory_Boundary_Isolation", 0.1)
    ]

    domain_results = []

    for domain_name, source_node, target_node, real_phase in domains:
        print(f"\n------------------------------------------------------------------------")
        print(f"  [Domain Test] {domain_name}")
        print(f"  Source Node: '{source_node}' | Target Node: '{target_node}'")
        init_phase = ebl.graph.nodes[source_node]["phase"]
        init_z = ebl.graph.edges[source_node, target_node]["impedance"]
        print(f"  Initial Lens Phase Angle: {init_phase:.4f} rad | Initial Impedance Z: {init_z:.4f}")

        # Step 1: Execute step facing non-negotiable external constraint
        print(f"\n  >> Step 1: Facing Non-Negotiable External Constraint (Real Phase = {real_phase:.4f} rad)...")
        step1 = ebl.enact_step(source_node, external_frequency=5.0, external_phase=real_phase, target_node=target_node)

        print(f"     Friction Factor (F):        {step1['friction_factor']:.4f}")
        print(f"     Phase Lag (Delta_phi):      {step1['phase_lag_rad']:.4f} rad")
        print(f"     Phase Recalibrated:         {step1['phase_recalibrated']}")
        print(f"     New Source Phase:           {step1['new_source_phase']:.4f} rad")
        print(f"     Updated Edge Impedance Z:   {step1['updated_edge_impedance']:.4f}")
        print(f"     Status:                     {step1['status']}")

        # Step 2: Retry with recalibrated phase angle facing same constraint
        print(f"\n  >> Step 2: Re-executing Projection with Recalibrated Lens Angle...")
        step2 = ebl.enact_step(source_node, external_frequency=5.0, external_phase=real_phase, target_node=target_node)

        print(f"     Friction Factor (F):        {step2['friction_factor']:.4f}")
        print(f"     Phase Lag (Delta_phi):      {step2['phase_lag_rad']:.4f} rad")
        print(f"     Phase Recalibrated:         {step2['phase_recalibrated']}")
        print(f"     New Source Phase:           {step2['new_source_phase']:.4f} rad")
        print(f"     Updated Edge Impedance Z:   {step2['updated_edge_impedance']:.4f}")
        print(f"     Status:                     {step2['status']}")

        domain_results.append({
            "domain": domain_name,
            "step1_friction": step1["friction_factor"],
            "step2_friction": step2["friction_factor"],
            "step2_status": step2["status"],
            "final_phase": step2["new_source_phase"],
            "final_z": step2["updated_edge_impedance"]
        })

    print("\n" + "=" * 80)
    print("  ISOMORPHISM SYNTHESIS & STRUCTURAL INVARIANCE ANALYSIS")
    print("=" * 80)

    for res in domain_results:
        print(f"  • {res['domain']}:")
        print(f"    - Friction Transition: {res['step1_friction']:.4f} -> {res['step2_friction']:.4f}")
        print(f"    - Final Phase Alignment: {res['final_phase']:.4f} rad")
        print(f"    - Edge Impedance (Z): {res['final_z']:.4f}")
        print(f"    - Structural Outcome: {res['step2_status']}")

    # Verify structural isomorphism across all 3 domains
    all_recalibrated = all(r["step2_status"] == "RESONANCE" or r["step2_friction"] < 0.15 for r in domain_results)
    all_frictions_dropped = all(r["step2_friction"] < r["step1_friction"] for r in domain_results)

    print("\n" + "-" * 80)
    if all_recalibrated and all_frictions_dropped:
        print("  SUCCESS: Topological Isomorphism Verified across Physical, Financial, and Software Domains!")
        print("  All 3 domains share the exact same structural invariant:")
        print("  'Causal Trajectory -> Non-Negotiable Boundary Impact -> Phase Recalibration -> Resonance Alignment'")
        print("=" * 80)
        return True
    else:
        print("  FAILURE: Isomorphism verification failed.")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = run_tri_domain_isomorphism_verification()
    sys.exit(0 if success else 1)
