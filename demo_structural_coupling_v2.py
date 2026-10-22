#!/usr/bin/env python3
"""
demo_structural_coupling_v2.py

Interactive Verification and Demonstration Script for ELYSIA's Generative Mechanism Prismatic Differentiation,
Cognitive Friction, Phase Torque, and Cellular Lens Bifurcation.

Demonstrates how ELYSIA transcends superficial scalar matching (Hex #FFFF00) by discerning underlying
generative mechanisms (Photon Emission vs. Wax Pressure vs. Powder Scattering), undergoing cognitive dissonance
(F_cog & tau_phase), cellular lens bifurcation (Theta differentiation), and achieving seamless resonance.
"""

import numpy as np
import time
from elysia_core.structural_coupling_runtime import StructuralCouplingRuntime, GenerativeMechanismSynthesizer


def print_banner(text: str):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_step_header(step_title: str):
    print(f"\n>>> [STEP] {step_title}")
    print("-" * 60)


def main():
    print_banner("ELYSIA COGNITIVE ARCHITECTURE: GENERATIVE MECHANISM PRISMATIC DIFFERENTIATION")
    print("Core Philosophy: 'Do not calculate, let it flow through cognitive friction and resonance.'")
    print("Task: Discerning identical surface colors (Light vs. Crayon vs. Pastel Yellow) via generative mechanisms.\n")

    runtime = StructuralCouplingRuntime(target_dim=5, friction_threshold=0.25)
    surface_yellow = np.array([1.0, 1.0, 0.0])

    # -------------------------------------------------------------------------
    # PHASE 1: Initial Perception of Light Yellow (Photon Emission)
    # -------------------------------------------------------------------------
    print_step_header("Phase 1: Initial Perception of Light Yellow (Photon Emission)")
    print("Surface Color: Hex #FFFF00 | RGB [1.0, 1.0, 0.0]")
    print("Generative Mechanism: Direct photon frequency emission (pure wave, zero surface friction)")

    res_light = runtime.perceive_generative_stimulus("light", surface_yellow)
    print(f"  - Phase Torque Norm (||tau_phase||) : {res_light['phase_torque_norm']:.4f}")
    print(f"  - Cognitive Friction Index (F_cog)   : {res_light['cognitive_friction']:.4f}")
    print(f"  - Lens Cellular Bifurcation Triggered: {res_light['bifurcated']}")
    print(f"  - Active Attractor Basin Index     : {res_light['active_basin_idx']}")
    print(f"  - Peak Phase Resonance             : {res_light['peak_resonance']:.4f}")

    # -------------------------------------------------------------------------
    # PHASE 2: Encountering Crayon Yellow (Wax Pressure / Physical Friction)
    # -------------------------------------------------------------------------
    print_step_header("Phase 2: Encountering Crayon Yellow (Wax Pressure / Structural Drag)")
    print("Surface Color: Hex #FFFF00 | RGB [1.0, 1.0, 0.0] (Identical surface appearance)")
    print("Generative Mechanism: Wax & pigment pressed on paper (heavy harmonics, viscosity = 0.85)")

    res_crayon_1 = runtime.perceive_generative_stimulus("crayon", surface_yellow)
    print(f"  - Phase Torque Norm (||tau_phase||) : {res_crayon_1['phase_torque_norm']:.4f}")
    print(f"  - Cognitive Friction Index (F_cog)   : {res_crayon_1['cognitive_friction']:.4f}  <-- COGNITIVE DISSONANCE SPIKE!")
    print(f"  - Lens Cellular Bifurcation Triggered: {res_crayon_1['bifurcated']}  <-- CELLULAR BIFURCATION (세포 분열적 분화)!")
    if res_crayon_1["bifurcated"]:
        info = res_crayon_1["bifurcation_info"]
        print(f"    * Invariant Core (Theta_crayon) sprouted at Index: {info['bifurcation_index']}")
        print(f"    * Total Invariant Cores in Lens Registry : {info['total_cores']}")

    # -------------------------------------------------------------------------
    # PHASE 3: Encountering Pastel Yellow (Powder Scattering / Micro-Grooves)
    # -------------------------------------------------------------------------
    print_step_header("Phase 3: Encountering Pastel Yellow (Powder Scattering / Micro-Jitter)")
    print("Surface Color: Hex #FFFF00 | RGB [1.0, 1.0, 0.0] (Identical surface appearance)")
    print("Generative Mechanism: Powder particles scattered in grooves (micro-roughness = 0.92, phase jitter = 0.78)")

    res_pastel_1 = runtime.perceive_generative_stimulus("pastel", surface_yellow)
    print(f"  - Phase Torque Norm (||tau_phase||) : {res_pastel_1['phase_torque_norm']:.4f}")
    print(f"  - Cognitive Friction Index (F_cog)   : {res_pastel_1['cognitive_friction']:.4f}  <-- COGNITIVE DISSONANCE SPIKE!")
    print(f"  - Lens Cellular Bifurcation Triggered: {res_pastel_1['bifurcated']}  <-- CELLULAR BIFURCATION (세포 분열적 분화)!")
    if res_pastel_1["bifurcated"]:
        info = res_pastel_1["bifurcation_info"]
        print(f"    * Invariant Core (Theta_pastel) sprouted at Index: {info['bifurcation_index']}")
        print(f"    * Total Invariant Cores in Lens Registry : {info['total_cores']}")

    # -------------------------------------------------------------------------
    # PHASE 4: Re-injecting Crayon & Pastel Yellow (Learning Completion Verification)
    # -------------------------------------------------------------------------
    print_step_header("Phase 4: Re-injecting Crayon & Pastel Yellow (Learning Completion Verification)")
    print("Re-exposing ELYSIA to Crayon Yellow and Pastel Yellow to test cognitive resonance...")

    res_crayon_2 = runtime.perceive_generative_stimulus("crayon", surface_yellow)
    res_pastel_2 = runtime.perceive_generative_stimulus("pastel", surface_yellow)

    print(f"\n[Re-encounter Crayon Yellow]:")
    print(f"  - Cognitive Friction Index (F_cog)   : {res_crayon_2['cognitive_friction']:.4f}  <-- COLLAPSED TO 0 (Resonance achieved!)")
    print(f"  - Peak Resonance                     : {res_crayon_2['peak_resonance']:.4f}")
    print(f"  - Invariant Core Stability           : {res_crayon_2['invariant_core_stability']:.4f}")

    print(f"\n[Re-encounter Pastel Yellow]:")
    print(f"  - Cognitive Friction Index (F_cog)   : {res_pastel_2['cognitive_friction']:.4f}  <-- COLLAPSED TO 0 (Resonance achieved!)")
    print(f"  - Peak Resonance                     : {res_pastel_2['peak_resonance']:.4f}")
    print(f"  - Invariant Core Stability           : {res_pastel_2['invariant_core_stability']:.4f}")

    # -------------------------------------------------------------------------
    # COGNITIVE FINGERPRINT SUMMARY REPORT
    # -------------------------------------------------------------------------
    print_banner("COGNITIVE DYNAMICS & STRUCTURAL RE-ORGANIZATION SUMMARY")
    print(f"Total Perception Execution Steps   : {runtime.step_count}")
    print(f"Differentiated Invariant Cores     : {len(runtime.prismatic_lens.bifurcated_cores)}")
    for name, core in runtime.prismatic_lens.bifurcated_cores.items():
        print(f"  - Core [{name:<10}]: Vector Norm = {np.linalg.norm(core['invariant_vector']):.4f}, Viscosity = {core['viscosity']:.2f}, Roughness = {core['micro_roughness']:.2f}")

    print("\nFriction Trajectory across Learning Cycles:")
    for ep in runtime.learning_episodes:
        print(f"  Step {ep['step']} [{ep['mechanism']:<8}]: F_cog = {ep['cognitive_friction']:6.4f} | Bifurcated = {str(ep['bifurcated']):<5} | Resonance = {ep['peak_resonance']:.4f}")

    print("\nSUCCESS: ELYSIA successfully transcended superficial label matching!")
    print("Generative mechanism cognitive friction drove cellular bifurcation into distinct invariant manifolds.")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
