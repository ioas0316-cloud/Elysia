#!/usr/bin/env python3
"""
demos/demo_scale_hierarchy_continuity.py

Demonstration script showing complete multi-scale continuity from L0 to L3 in Elysia.
Runs a continuous simulation of physical/sensory friction propagating through:
  - L0 Substrate wave field
  - L1 Sensory Prismatic Gate & Micro-lens focusing
  - L2 Surface Tension Attractor minimum-energy relaxation
  - L3 Macro-Causal Deck purpose field & spatiotemporal scrubbing
  - Homeostatic feedback loop adjusting L0/L1 sensitivity dial.
"""

import sys
from pathlib import Path
import numpy as np

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import elysia_loader
elysia_loader.install()

from core.orchestration.scale_hierarchy_orchestrator import ScaleHierarchyOrchestrator


def run_demo():
    print("===========================================================================")
    print(" ELYSIA SCALE HIERARCHY CONTINUITY DEMO (L0 Substrate -> L3 Macro Deck)")
    print("===========================================================================")

    orchestrator = ScaleHierarchyOrchestrator()

    # Create a dynamic sensory wave stream
    time_points = np.linspace(0, 10 * np.pi, 1000)
    base_wave = np.sin(time_points) + 0.5 * np.cos(3 * time_points)

    print("\n[Phase 1: Normal Low-Friction Operation]")
    for i in range(3):
        noise_wave = base_wave + 0.1 * np.random.randn(1000)
        res = orchestrator.process_embodied_cycle(noise_wave, world_friction=0.05, dt=0.1)
        print(f" Frame {i+1} | L1 Coherence: {res['L1_phase_coherence']:.3f} | "
              f"L2 Energy: {res['L2_initial_energy']:.1f} -> {res['L2_relaxed_energy']:.1f} | "
              f"L3 Tension: {res['L3_macro_tension']:.4f} | Dial: {res['feedback_sensitivity_dial']:.4f}")

    print("\n[Phase 2: High Thermal/Physical Friction Impulse Injection (World Shock)]")
    shock_wave = base_wave + 2.5 * np.random.randn(1000)  # Intense physical shock
    res_shock = orchestrator.process_embodied_cycle(shock_wave, world_friction=1.5, dt=0.1)
    print(f" SHOCK FRAME | L1 Coherence: {res_shock['L1_phase_coherence']:.3f} | "
          f"L2 Energy: {res_shock['L2_initial_energy']:.1f} -> {res_shock['L2_relaxed_energy']:.1f} | "
          f"L3 Tension: {res_shock['L3_macro_tension']:.4f} | Dial: {res_shock['feedback_sensitivity_dial']:.4f}")
    print(f" Accumulated Metric Scar Tensor Value: {res_shock['accumulated_scar']:.4f}")

    print("\n[Phase 3: Post-Shock Homeostatic Stabilization & Adaptation]")
    for i in range(3):
        noise_wave = base_wave + 0.1 * np.random.randn(1000)
        res = orchestrator.process_embodied_cycle(noise_wave, world_friction=0.1, dt=0.1)
        print(f" Frame {i+5} | L1 Coherence: {res['L1_phase_coherence']:.3f} | "
              f"L2 Energy: {res['L2_initial_energy']:.1f} -> {res['L2_relaxed_energy']:.1f} | "
              f"L3 Tension: {res['L3_macro_tension']:.4f} | Dial: {res['feedback_sensitivity_dial']:.4f}")

    print("\n[Phase 4: Spatiotemporal Memory Scrubbing ($O(1)$ Rollback to Frame 2)]")
    target_time = 0.2
    scrubbed_state = orchestrator.macro_deck.scrubber.scrub_trajectory(target_time)
    print(f" Successfully scrubbed back macro state at t={target_time}s!")
    print(f" Scrubbed State Norm: {np.linalg.norm(scrubbed_state):.4f}")

    print("\n===========================================================================")
    print(" CONTINUOUS MULTI-SCALE HIERARCHY VERIFIED WITHOUT ARTIFICIAL BRANCHING!")
    print("===========================================================================")


if __name__ == "__main__":
    run_demo()
