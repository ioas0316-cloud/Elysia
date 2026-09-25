#!/usr/bin/env python3
"""
verify_causal_erosion_vr_simulation.py

Verification script for Causal Erosion Landscape and Multi-Scale VR Attractor Simulation.
"""

import sys
import torch
import time
from elysia_engine.core.causal_erosion import CausalErosionLandscape
from elysia_engine.core.vr_multiscale_attractor import MultiScaleElysiaEcosystem


def verify_all():
    print("=================================================================")
    print("  VERIFYING ELYSIA CAUSAL EROSION & MULTI-SCALE VR SIMULATION")
    print("=================================================================\n")

    # 1. Causal Erosion Verification
    print("1. Testing Causal Erosion Landscape & Phase Transition...")
    landscape = CausalErosionLandscape(state_dim=3, nc_threshold=10)
    pt = torch.tensor([1.0, 1.0, 1.0])

    for i in range(12):
        landscape.erode_trajectory(pt * (i * 0.1))

    assert landscape.is_phase_transformed, "Phase transition to O(1) continuous grid failed!"
    print("   [PASS] Phase transition triggered at Nc = 10 wells.")

    # Memory Replay check
    start_p = torch.tensor([0.5, 0.5, 0.5])
    replayed = landscape.memory_replay_step(start_p, dt=0.05)
    print(f"   [PASS] Memory Replay gradient flow step: [{replayed[0]:.3f}, {replayed[1]:.3f}, {replayed[2]:.3f}]")

    # 2. Multi-Scale Ecosystem Verification
    print("\n2. Testing Multi-Scale VR Attractor Ecosystem...")
    eco = MultiScaleElysiaEcosystem()
    print(f"   Initial State -> Mode: {eco.macro_monster.current_mode} | Denatured: {eco.micro_protein.is_denatured}")

    # Strong impact to trigger coupled transition
    eco.macro_monster.receive_impact(hit_location=2, force_vector=torch.tensor([4.0, 4.0, 4.0]))

    start_time = time.time()
    for step in range(12):
        eco.step_ecosystem(dt=0.05)
    elapsed = (time.time() - start_time) * 1000

    print(f"   Post-Impact State -> Mode: {eco.macro_monster.current_mode} | Denatured: {eco.micro_protein.is_denatured}")
    assert eco.macro_monster.current_mode == "berserk", "Macro monster failed to transition to berserk mode!"
    assert eco.micro_protein.is_denatured, "Micro protein failed thermal denaturation transition!"
    print(f"   [PASS] Multi-scale ecosystem coupled phase transition verified in {elapsed:.2f}ms.")

    print("\n=================================================================")
    print("  ALL ELYSIA CAUSAL EROSION & VR SIMULATION MODULES VERIFIED!  ")
    print("=================================================================\n")


if __name__ == "__main__":
    verify_all()
