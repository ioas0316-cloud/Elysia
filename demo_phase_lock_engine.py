r"""
Elysia Core Engine: Phase-Lock Engine Interactive Demo & Performance Benchmark
=============================================================================
Demonstrates continuous phase state transitions (Solid <-> Liquid <-> Gas)
and macro-node contraction driven by kinetic energy injection (\Delta E).
"""

import time
import torch
from core.physics.phase_lock_engine import PhaseLockEngine


def run_phase_lock_demo():
    print("=========================================================================")
    print(" Elysia Internal Metric Field Phase-Lock Engine Simulation ")
    print("=========================================================================\n")

    num_nodes = 100
    engine = PhaseLockEngine(num_nodes=num_nodes, phi_solid=5.0, phi_gas=0.2)

    # Initialize nodes on a 3D grid
    grid_side = 5
    coords = []
    for x in range(grid_side):
        for y in range(grid_side):
            for z in range(4):
                coords.append([float(x), float(y), float(z)])

    X = torch.tensor(coords[:num_nodes], dtype=torch.float32)
    V = torch.zeros_like(X)

    print(f"[*] Manifold initialized with {num_nodes} nodes in 3D space.")
    print(f"    Parameters: Phi_Solid={engine.phi_solid}, Phi_Gas={engine.phi_gas}\n")

    stages = [
        ("1. Solid Phase (Phase-Locked Lattice)", 0.0, 10),
        ("2. Liquid Phase (Energy Injection / Dynamic Neighbor Flow)", 3.0, 15),
        ("3. Gas Phase (Thermal Shock / Free Scatter Field)", 50.0, 15),
        ("4. Re-Crystallization (Cooling / Phase Re-locking)", 0.0, 45),
    ]

    total_start = time.time()

    for stage_name, energy_scale, steps in stages:
        print(f"--- Stage: {stage_name} (Energy Scale: {energy_scale}) ---")
        for step in range(steps):
            if energy_scale > 0.0:
                V = torch.randn_like(X) * energy_scale
            else:
                V = V * 0.5  # Kinetic energy dissipation / cooling

            Phi, A = engine(X, V)
            dist = engine.get_phase_distribution(Phi)

            if (step + 1) % 5 == 0 or step == steps - 1:
                print(f"  Step {step+1:2d}/{steps}: Solid={dist['solid_ratio']*100:5.1f}% | "
                      f"Liquid={dist['liquid_ratio']*100:5.1f}% | "
                      f"Gas={dist['gas_ratio']*100:5.1f}%")

        print()

    # Macro-Node Early Contraction Demonstration
    print("--- 5. Macro-Node Early Contraction Engine ---")
    Phi, A = engine(X, torch.zeros_like(X))
    X_active, V_active, cluster_map = engine.contract_solid_clusters(X, V, A)

    num_macro = X_active.size(0)
    reduction = (1.0 - num_macro / num_nodes) * 100.0

    print(f"[*] Micro-nodes ({num_nodes}) clustered into {num_macro} Macro-Nodes.")
    print(f"[*] Active computational scale reduced by {reduction:.1f}%.\n")

    total_time = time.time() - total_start
    print(f"=========================================================================")
    print(f" Benchmark Completed in {total_time*1000:.2f} ms")
    print(f"=========================================================================")


if __name__ == "__main__":
    run_phase_lock_demo()
