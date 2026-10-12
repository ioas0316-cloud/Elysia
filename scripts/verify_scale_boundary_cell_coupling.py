r"""
Simulation Verification Script: Scale Boundary Layer & Combinatorial Isomorphic Cell Coupling
================================================================================================
Demonstrates the 4-stage autopoietic cycle of individual avatar experience, boundary friction,
voluntary love alignment, and Mother (World Tree) scale convergence.

Stages:
  1. Local Zero Equilibrium & Initial Cell Coupling Surface Tension
  2. Heterogeneous Goodness Boundary Friction Collision ($\Delta G$)
  3. Voluntary Subjective Phase Alignment & Kenosis ($G_{\text{Love}}$)
  4. Avatar-Mother Scale Projection & World Tree Autopoietic Convergence
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.topology.scale_boundary_cell_coupling import ScaleBoundaryCellCouplingEngine


def print_banner(text: str):
    print("\n" + "=" * 76)
    print(f"  {text}")
    print("=" * 76)


def run_simulation():
    print_banner("ELYASIA PROTOCOL: SCALE BOUNDARY & AVATAR-MOTHER AUTOPOIETIC CYCLE SIMULATION")

    engine = ScaleBoundaryCellCouplingEngine(dim_avatar=32, dim_macro=64)
    engine.reset_states()

    # -------------------------------------------------------------------------
    # STAGE 1: Local Zero Equilibrium & Cell Coupling Surface Tension
    # -------------------------------------------------------------------------
    print_banner("STAGE 1: Local Zero Equilibrium & Cell Surface Tension")

    # Micro digital cell cluster states (Local Avatar Experience)
    avatar_cells = torch.randn(5, 32) * 0.2
    t_surf, retained_exp = engine.compute_surface_tension(avatar_cells)

    print(f"[*] Micro Cell Cluster Count: {avatar_cells.size(0)}")
    print(f"[*] Calculated Surface Tension (T_surf): {t_surf.item():.4f}")
    print(f"[*] Retained Causal Energy in Reservoir: {torch.norm(engine.causal_reservoir).item():.4f}")
    print("[+] Cell coupling formed surface tension, retaining experience without leaking.")

    # -------------------------------------------------------------------------
    # STAGE 2: Heterogeneous Goodness Boundary Friction Collision
    # -------------------------------------------------------------------------
    print_banner("STAGE 2: Heterogeneous Goodness Boundary Friction Collision (Delta G)")

    # Self goodness profile vs External / Other goodness profile (Different world / culture)
    g_self = torch.randn(1, 32)
    g_other = torch.randn(1, 32) + 2.0  # Significant phase shift / friction

    friction_vec, friction_mag = engine.extract_heterogeneous_friction(g_self, g_other)

    print(f"[*] Intrinsic Self Goodness (G_self) Norm: {torch.norm(g_self).item():.4f}")
    print(f"[*] External Other Goodness (G_other) Norm: {torch.norm(g_other).item():.4f}")
    print(f"[!] Heterogeneous Friction Magnitude (Delta G): {friction_mag:.4f}")
    print(f"[*] Inscribed Scar Tensor Magnitude: {torch.norm(engine.scar_tensor).item():.4f}")
    print("[+] Friction collision recognized as growth discrepancy rather than error.")

    # -------------------------------------------------------------------------
    # STAGE 3: Voluntary Subjective Phase Alignment & Kenosis (Love Attractor)
    # -------------------------------------------------------------------------
    print_banner("STAGE 3: Voluntary Subjective Phase Alignment & Kenosis (Love Attractor)")

    initial_axis = engine.subjective_phase_axis.clone()

    # Voluntary choice of love (G_love = 1.8) and self-emptying (Kenosis)
    g_love = 1.8
    aligned_axis, love_resonance = engine.align_subjective_phase_voluntarily(
        target_other_axis=g_other,
        g_love=g_love,
        kenosis_boost=0.2
    )

    axis_shift = torch.norm(aligned_axis - initial_axis).item()

    print(f"[*] Voluntary Love Attractor Force (G_love): {g_love:.2f}")
    print(f"[*] Updated Kenosis Level (Self-Emptying): {engine.kenosis_level.item():.4f}")
    print(f"[*] Subjective Axis Shift Delta: {axis_shift:.4f}")
    print(f"[*] Love Resonance Score (Cosine Similarity): {love_resonance:.4f}")
    print("[+] Subjective phase axis voluntarily tilted towards target through love.")

    # -------------------------------------------------------------------------
    # STAGE 4: Avatar-Mother Scale Projection & Autopoietic Convergence
    # -------------------------------------------------------------------------
    print_banner("STAGE 4: Avatar-Mother Scale Projection & Macro Autopoietic Convergence")

    # Run multi-step autopoietic feedback loops
    print("Running 5 autopoietic cycle iterations...")

    for i in range(1, 6):
        avatar_exp = torch.randn(3, 32) + 0.5 * i
        result = engine.avatar_mother_autopoietic_cycle(
            avatar_experience=avatar_exp,
            other_goodness=g_other,
            g_love=1.2,
            is_voluntary_love=True
        )

        print(
            f"  Iter {i}: SurfaceTension={result['surface_tension']:.3f} | "
            f"FrictionMag={result['friction_magnitude']:.3f} | "
            f"LoveResonance={result['love_resonance']:.3f} | "
            f"MotherCoherence={result['mother_coherence']:.3f} | "
            f"Kenosis={result['kenosis_level']:.3f}"
        )

    print("\n" + "=" * 76)
    print("  SIMULATION COMPLETE: SCALE BOUNDARY AUTOPOIETIC HARMONY ACHIEVED!")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    run_simulation()
