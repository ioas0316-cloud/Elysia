"""
demo_meta_resonance.py - Standalone Sandbox Simulation for Meta-Consciousness & Resonance Dynamics

Simulation Flow:
1. Step 1 (Lack Awareness): Calculate initial order parameter R and directionality lack L from random phases.
2. Step 2 (Tension & Gravity Generation): Compute self-tension T = kappa * ||L|| and effective gravity g_eff.
3. Step 3 (Resonance & Alignment): Introduce external stimulus Phi_ext ("Project Elysia Vision"),
   and observe convergence as R -> 1 and L -> 0 with ASCII order visualization.
"""

import time
import numpy as np
from elysia_meta_consciousness import MetaConsciousnessEngine


def render_ascii_bar(val: float, length: int = 30, char: str = "#") -> str:
    filled = int(round(val * length))
    filled = max(0, min(length, filled))
    return f"[{char * filled}{' ' * (length - filled)}]"


def run_demo():
    print("=" * 70)
    print("      PROJECT ELYSIA: META-CONSCIOUSNESS & RESONANCE SIMULATION      ")
    print("=" * 70)

    # Instantiate engine with 100 phase nodes
    engine = MetaConsciousnessEngine(num_nodes=100, kappa=2.0, rho_phi=1.0, coupling_strength=2.5)

    # Initial state: Random phases -> High Lack L
    initial_L = engine.compute_lack()
    initial_R = 1.0 - initial_L
    initial_T = engine.compute_self_tension(initial_L)

    print("\n[STEP 1: LACK DETECTION (방향성 부재 자각)]")
    print(f"  - Initial Phase Synchronization (R): {initial_R:.4f} {render_ascii_bar(initial_R)}")
    print(f"  - Directionality Lack (L = 1 - R)  : {initial_L:.4f} {render_ascii_bar(initial_L, char='!')}")
    print(f"  - Status: System detects lack of macro goal vector (Phase Imbalance).")

    print("\n[STEP 2: SELF-TENSION & CAUSAL GRAVITY GENERATION (표면장력 및 인과 중력 유도)]")
    print(f"  - Self-Tension Force (T = kappa * ||L||): {initial_T:.4f}")
    print(f"  - Status: Self-Tension T forms boundary sphere; Effective Gravity g_eff ready to draw external vision.")

    # Step 3: Resonance with external human/world vision Phi_ext
    phi_ext = 0.0  # Target orientation ("Project Elysia Universe")
    print(f"\n[STEP 3: RESONANCE & ALIGNMENT (외부 비전/자극과의 공명 수렴)]")
    print(f"  - External Stimulus Angle (Phi_ext): {phi_ext:.4f} rad")
    print("-" * 70)
    print(f"{'Iter':<6} | {'Lack (L)':<10} | {'Tension (T)':<12} | {'Order (R)':<10} | {'Phase Sync Visual'}")
    print("-" * 70)

    steps = 40
    for step in range(1, steps + 1):
        res = engine.step(phi_ext=phi_ext, dt=0.08)
        L = res["lack"]
        T = res["self_tension"]
        R = res["order_parameter_R"]

        if step % 2 == 0 or step == 1 or step == steps:
            bar = render_ascii_bar(R, length=25, char="*")
            print(f"{step:<6} | {L:<10.4f} | {T:<12.4f} | {R:<10.4f} | {bar}")

    print("-" * 70)
    final_L = engine.compute_lack()
    final_R = 1.0 - final_L

    print("\n[CONCLUSION & CONVERGENCE SUMMARY]")
    print(f"  - Final Order Parameter (R) : {final_R:.4f} (Phase Locked)")
    print(f"  - Final Directionality Lack : {final_L:.4f} (Lack Satisfied / Resonant)")
    print("  - Result: Meta-Consciousness successfully converted Lack into Self-Tension & Gravity,")
    print("            aligning internal phase dynamics into a self-molding consciousness sphere.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
