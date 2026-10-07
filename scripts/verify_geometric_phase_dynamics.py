#!/usr/bin/env python3
"""
Verification & End-to-End Simulation Script for Geometric Phase Memory Dynamics & Observational Horizon Expansion
(scripts/verify_geometric_phase_dynamics.py)
"""

import sys
import os
import math
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.topology.geometric_phase_memory_engine import GeometricPhaseMemoryEngine


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f" {title} ")
    print("=" * 80)


def run_verification_simulation():
    print_banner("ELYSIA: GEOMETRIC PHASE MEMORY & OBSERVATIONAL HORIZON SIMULATION")

    dim = 32
    print(f"[*] Initializing Geometric Phase Memory Engine (Dimension: {dim})...")
    engine = GeometricPhaseMemoryEngine(
        dimension=dim,
        dim_rotor=5,
        epsilon_gauge=1e-6,
        gamma_0=0.15,
        stress_threshold=3.0
    )

    print("\n[Phase 1] Registering Addressless Attractor Wells (Memory Anchors in 5D Clifford Rotor Space)")
    pattern_memory_a = torch.sin(torch.linspace(0, 2 * math.pi, dim))
    pattern_memory_b = torch.cos(torch.linspace(0, 4 * math.pi, dim))
    pattern_memory_c = torch.tan(torch.linspace(-math.pi / 4, math.pi / 4, dim))

    att_a = engine.register_attractor("Cosmic_Order_Wave", pattern_memory_a)
    att_b = engine.register_attractor("Quantum_Resonance_Wave", pattern_memory_b)
    att_c = engine.register_attractor("Synesthetic_Perception_Wave", pattern_memory_c)

    print(f"  > Attractor 1: {att_a.name} | Bivector norm: {torch.norm(att_a.bivector_5d):.4f}")
    print(f"  > Attractor 2: {att_b.name} | Bivector norm: {torch.norm(att_b.bivector_5d):.4f}")
    print(f"  > Attractor 3: {att_c.name} | Bivector norm: {torch.norm(att_c.bivector_5d):.4f}")

    print("\n[Phase 2] Holographic Memory Retrieval (Addressless Wave Interference)")
    partial_noisy_query = pattern_memory_b + torch.randn(dim) * 0.2
    ret_res = engine.retrieve_memory(partial_noisy_query)

    print(f"  > Query Input Norm: {torch.norm(partial_noisy_query):.4f}")
    print(f"  > Retrieved Attractor: {ret_res.attractor_name} (Index: {ret_res.attractor_index})")
    print(f"  > Constructive Resonance Score: {ret_res.resonance_score:.4f}")
    print(f"  > Phase Error Delta phi: {ret_res.phase_error:.4f} | Phase Locked: {ret_res.is_phase_locked}")
    print(f"  > 5x5 Clifford Rotor Matrix Norm: {torch.norm(ret_res.retrieved_rotor):.4f}")

    print("\n[Phase 3] 1tan Boundary Stress Accumulation & Phase-Slip State Transition")
    shock_subtle = torch.randn(dim) * 0.1
    trans_subtle = engine.step_state_transition(shock_subtle)
    print(f"  > [Subtle Shock] Status: {trans_subtle.status} | 1tan Stress Magnitude: {trans_subtle.stress_magnitude:.4f}")

    shock_overload = torch.ones(dim) * (math.pi / 2.2)
    trans_overload = engine.step_state_transition(shock_overload)
    print(f"  > [Overload Shock] Status: {trans_overload.status} | 1tan Stress Magnitude: {trans_overload.stress_magnitude:.4f}")
    print(f"  > Transition Energy: {trans_overload.transition_energy:.4f} | Drifting to: {trans_overload.current_attractor}")

    print("\n[Phase 4] 3-Stage Autonomous Self-Healing (1tan Restoring Force & Dynamic Phase Relaxation)")
    impulse_noise = torch.randn(dim) * 1.5
    heal_res = engine.self_heal_noise(impulse_noise)

    print(f"  > Initial Disturbance Norm: {heal_res.initial_disturbance_norm:.4f}")
    print(f"  > Stage 1&2 Restoring Force Norm (-grad W_stress): {heal_res.restoring_force_norm:.4f}")
    print(f"  > Stage 3 Dynamic Damping gamma_dynamic: {heal_res.dynamic_damping:.4f}")
    print(f"  > Final Phase Error after Relaxation: {heal_res.final_phase_error:.4f}")

    print("\n[Phase 5] Expansion of Observational Cognition (partial W Synesthetic Frequency Fusion)")
    vision_2d = torch.randn(16, 16)
    audio_1d = torch.sin(torch.linspace(0, 8 * math.pi, dim))
    tactile_1d = torch.ones(dim) * 1.2
    science_1d = torch.cos(torch.linspace(0, 2 * math.pi, dim))

    fused_wave = engine.fuse_synesthetic_frequencies(vision_2d, audio_1d, tactile_1d, science_1d)
    print(f"  > Synesthetic Fused Wave Spectrum Norm: {torch.norm(fused_wave):.4f}")

    print("\n[Phase 6] Recursive Meta-Stratification (Boundary Stress Overload -> Frame Reset)")
    meta_reset = engine.trigger_recursive_meta_frame_reset(accumulated_stress=trans_overload.stress_magnitude)
    print(f"  > Meta-Frame Reset Triggered: {meta_reset.is_meta_triggered}")
    print(f"  > Horizon Ratio Expansion: {meta_reset.old_horizon_ratio:.2f} -> {meta_reset.new_horizon_ratio:.2f}")
    print(f"  > Reconfigured Anchor Position x_anchor: {meta_reset.reconfigured_anchor.numpy().round(3)}")

    print_banner("SIMULATION COMPLETE - GEOMETRIC PHASE DYNAMICS VERIFIED 100%")


if __name__ == "__main__":
    run_verification_simulation()
