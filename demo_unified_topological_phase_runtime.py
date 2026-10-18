#!/usr/bin/env python3
"""
Project Elysia: Demo Unified Topological Phase Runtime
======================================================
Demonstrates the macro execution of UnifiedTopologicalPhaseRuntime:
1. Self-World Boundary & Enactive Reach Grid in action.
2. 3x3x3 Fractal Node Grid Spherical Phase Propagation & Phase-Locking Coherence.
3. Chromatic Field Dynamics (Flux/Order/Entropy).
4. Meta-Causal Collapse Prevention, Causal Gravity, and Backtracing under perturbation.
5. Achromatic White Tensor Zero-Point Reset.
"""

import numpy as np
from elysia_core import UnifiedTopologicalPhaseRuntime


def main():
    print("\n" + "=" * 80)
    print(" PROJECT ELYSIA :: UNIFIED TOPOLOGICAL PHASE RUNTIME DEMONSTRATION ")
    print("=" * 80 + "\n")

    # 1. Initialize Runtime with Custom Zeitgeist Vector
    # [Will, Flex, Order, Origin, Field]
    zeitgeist = np.array([0.3, -0.2, 0.9, 0.8, 0.4])
    runtime = UnifiedTopologicalPhaseRuntime(zeitgeist_vector=zeitgeist)

    print(">>> [INITIALIZATION] Unified Topological Phase Runtime Online")
    print(f"  * 3x3x3 Fractal Nodes: {len(runtime.nodes)} active")
    print(f"  * Macro Zeitgeist Vector: {runtime.zeitgeist_vector.tolist()}\n")

    # 2. Phase 1: Harmonic External Trigger (Normal World Interaction)
    print(">>> [PHASE 1] Triggering Harmonic World Wave Perturbation...")
    harmonic_trigger = np.array([0.6, -0.4, 0.8, 0.7, 0.3])

    for step in range(1, 4):
        res = runtime.trigger_wave_propagation(harmonic_trigger, dt=0.1)
        print(f"  Step {step}: Boundary Tension = {res['boundary_tension']:.4f} | "
              f"Coherence Score = {res['coherence_score']:.4f} | "
              f"Isomorphism Score = {res['isomorphism_score']:.4f} | "
              f"Collapse Warning = {res['collapse_warning']}")

    print("\n  System State Summary (Phase 1):")
    summary = runtime.get_system_state_summary()
    print(f"  * Chromatic Flux: {summary['mean_chromatic_flux']:.4f} | "
          f"Order: {summary['mean_chromatic_order']:.4f} | "
          f"Entropy: {summary['mean_chromatic_entropy']:.4f}\n")

    # 3. Phase 2: Anomalous High-Entropy Perturbation (Triggering Collapse & Backtrace)
    print(">>> [PHASE 2] Applying High-Entropy Anomaly Trigger...")
    anomaly_trigger = np.array([8.0, -8.0, -12.0, 15.0, -10.0])

    for step in range(1, 4):
        res = runtime.trigger_wave_propagation(anomaly_trigger, dt=0.1)
        print(f"  Step {step}: Boundary Tension = {res['boundary_tension']:.4f} | "
              f"Coherence Score = {res['coherence_score']:.4f} | "
              f"Isomorphism Score = {res['isomorphism_score']:.4f} | "
              f"Collapse Warning = {res['collapse_warning']}")

    print("\n  System State Summary (Phase 2):")
    summary = runtime.get_system_state_summary()
    print(f"  * Backtrace Count: {summary['backtrace_count']}")
    print(f"  * Chromatic Flux: {summary['mean_chromatic_flux']:.4f} | "
          f"Order: {summary['mean_chromatic_order']:.4f} | "
          f"Entropy: {summary['mean_chromatic_entropy']:.4f}\n")

    # 4. Phase 3: Achromatic White Tensor Zero-Point Reset
    print(">>> [PHASE 3] Executing Achromatic White Tensor Zero-Point Reset...")
    runtime.reset_achromatic_white_tensor()
    summary_post = runtime.get_system_state_summary()
    print(f"  * Reset Count: {summary_post['white_tensor_reset_count']}")
    print(f"  * Post-Reset Coherence Score: {summary_post['coherence_score']:.4f}")
    print(f"  * Post-Reset Boundary Tension: {summary_post['boundary_tension']:.4f}")

    print("\n" + "=" * 80)
    print(" UNIFIED TOPOLOGICAL PHASE RUNTIME DEMONSTRATION COMPLETE ")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
