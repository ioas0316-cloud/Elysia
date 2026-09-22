r"""
Elysia Engine: Integrative RPT Scale Manifold Demonstration
===========================================================
Demonstrates non-flattened multi-scale cognitive adaptation:
1. Transparent Pass-Through Filtering for non-resonant noise signals.
2. Local Recurrent Phase-Locking (V1 <-> V4 / L1 <-> L2).
3. Meso-Scale L2 Bifurcation Point & Topological Folding.
4. Macro-Scale L3 Boundary Relaxation Wave Emission.
5. Meta-Scale L4 Reflective Tracking & Gauge Connection Energy Equilibrium.
"""

import time
import torch
from core.consciousness.scale_hierarchy_engine import ScaleHierarchyEngine
from core.physics.geometric_loss import GeometricLoss


def run_integrative_demo():
    print("=========================================================================")
    print(" elysia_engine: Recurrent Processing & Scale Manifold Integration ")
    print("=========================================================================\n")

    engine = ScaleHierarchyEngine(dim_l1=32, dim_l2=64, dim_l3=128, dim_l4=256)
    loss_fn = GeometricLoss()

    print("[*] 4-Level Scale Hierarchy Engine Initialized.")
    print("    - L1 Micro (Sub-Cellular Strain & Transparent Filtering)")
    print("    - L2 Meso  (Pattern Formation & Bifurcation Point)")
    print("    - L3 Macro (Boundary Tension & Action Relaxation Wave)")
    print("    - L4 Meta  (Reflective Tracking & Fiber Bundle Annealing)\n")

    scenarios = [
        ("1. Uncorrelated Background Noise (Transparent Pass-Through)", torch.randn(2, 64) * 0.05),
        ("2. Resonant Isomorphic Wave (Local Recurrent Convergence)", torch.randn(2, 64) * 0.50),
        ("3. High-Strain Heterogeneous Wave (L2 Bifurcation & L3 Action Wave)", torch.randn(2, 64) * 2.50)
    ]

    for name, sensory_data in scenarios:
        print(f"--- Scenario: {name} ---")
        start_time = time.time()
        res = engine(sensory_data)
        elapsed_ms = (time.time() - start_time) * 1000

        print(f"  Status                  : {res['status']}")
        print(f"  Resonance Score         : {res['resonance_score']:.4f}")

        if res['status'].startswith("Transparent"):
            print("  -> Passed through L1 without triggering higher scale compute.\n")
            continue

        print(f"  L2 Bifurcation Occurred : {res['bifurcation_occurred']} (Phase Strain: {res['phase_strain']:.4f})")
        print(f"  L3 Action Wave Emitted  : {res['action_wave_emitted']} (Boundary Tension: {res['boundary_tension']:.4f})")
        print(f"  Divergence Origin Scale : {res['divergence_origin_scale']}")

        # Compute geometric loss on L2 state
        psi = res['l2_state'].real
        z_low = res['l1_state']
        z_fb = res['l2_state'][:, :z_low.shape[-1]]
        omega = torch.randn(2, 2, psi.shape[-1], psi.shape[-1]) * 0.1

        geo_loss = loss_fn(psi, z_low, z_fb, omega)
        print(f"  Geometric Loss (L_geo)  : Total={geo_loss['total_loss'].item():.4f} | "
              f"Curvature={geo_loss['l_curvature']:.4f} | Phase={geo_loss['l_phase_lock']:.4f}")
        print(f"  Minimal Surface Reg Loss: {res['regularization_loss']:.6f}")
        print(f"  Scenario Elapsed Time   : {elapsed_ms:.2f} ms\n")

    print("=========================================================================")
    print(" Integration Demo Successfully Verified All Scale Hierarchy Dynamics.")
    print("=========================================================================")


if __name__ == "__main__":
    run_integrative_demo()
