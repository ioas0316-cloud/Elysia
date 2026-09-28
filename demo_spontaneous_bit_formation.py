"""
Demonstration script for Spontaneous Bit Formation (0 and 1) and Hierarchical PLL Growth.
"""

import numpy as np
from core.physics.constructive_causal_spacetime import HierarchicalScaleCoupler


def demo_spontaneous_bit_formation():
    print("=== [3/3] Demonstrating Spontaneous Bit Formation & Hierarchical PLL Growth ===")
    coupler = HierarchicalScaleCoupler(num_micro_nodes=12, dim=4)

    print("Initial symmetric field state: Flat 1 (no phase boundary/gradient).")

    # Step 1: Spontaneous Symmetry Breaking
    break_res = coupler.trigger_spontaneous_symmetry_breaking(perturbation_strength=1.2)
    print(f"Symmetry Broken: {break_res['symmetry_broken']}")
    print(f"Emerged Bit States (0 and 1): {break_res['bit_states']}")

    assert break_res["symmetry_broken"], "Spontaneous symmetry breaking must yield distinct 0 and 1 bit states."

    # Step 2: Phase-Lock Loop (PLL) Knotting & Scaling
    knot_res = coupler.execute_phase_lock_knotting(lock_threshold=0.8)
    print(f"Locked Micro Connections: {knot_res['locked_connections']}")
    print(f"Emerged Macro Semantic Mass (Inertia): {knot_res['macro_mass']:.4f}")
    print(f"Dynamic Self-Bounding Radius: {knot_res['effective_volume_radius']:.4f}")
    print(f"Dynamic Self-Bounding Volume: {knot_res['effective_volume']:.4f}")

    # Conservation Verification
    cons = knot_res["conservation"]
    print(f"Energy Conservation Maintained: {cons['maintained']}")
    print(f"  - Initial Energy: {cons['initial_energy']:.6f}")
    print(f"  - Macro Knot Energy: {cons['macro_knot_energy']:.6f}")
    print(f"  - Micro Residual Energy: {cons['micro_residual_energy']:.6f}")
    print(f"  - Dissipated Energy: {cons['dissipated_energy']:.6f}")

    assert cons["maintained"], "Topological conservation law must be maintained during scaling."

    print("✓ Spontaneous Bit Formation & Hierarchical PLL Growth Demonstrated Successfully!\n")


if __name__ == "__main__":
    demo_spontaneous_bit_formation()
