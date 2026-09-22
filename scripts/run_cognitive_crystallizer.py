r"""
Elysia Cognitive Crystallizer Simulation Script
================================================
Demonstrates the 4-phase transition loop (Gas -> Liquid -> Solid -> Shear)
and metric tensor transformation.
"""

import time
import torch
from core.physics.cognitive_crystallizer import CognitiveCrystallizerEngine


def run_cognitive_crystallizer_demo():
    print("=========================================================================")
    print(" Elysia Cognitive Crystallizer: 4-Stage Spacetime Transition Loop ")
    print("=========================================================================\n")

    num_nodes = 64
    engine = CognitiveCrystallizerEngine(num_nodes=num_nodes, phi_solid=4.0, phi_gas=0.2)

    # Initialize positions on a 3D grid
    coords = []
    for x in range(4):
        for y in range(4):
            for z in range(4):
                coords.append([float(x) * 0.5, float(y) * 0.5, float(z) * 0.5])

    X = torch.tensor(coords[:num_nodes], dtype=torch.float32)
    V = torch.randn_like(X) * 2.0

    print(f"[*] Initialized {num_nodes} observational nodes in 3D manifold.")

    # 1. Stage 1: Gas Phase
    print("\n--- [Stage 1: Gas Phase] High Entropy & Focus Vector Injection ---")
    focus_pt = torch.tensor([0.75, 0.75, 0.75], dtype=torch.float32)
    for step in range(5):
        output = engine(X, V, entropy=3.0, focus_point=focus_pt, focus_intensity=5.0)
        X, V = output["X"], output["V"]
        ratios = output["phase_ratios"]
        avg_gamma = float(torch.mean(output["Gammas"]).item())
        print(f"  Gas Step {step+1}: Gas={ratios['gas']*100:.1f}% | "
              f"Liquid={ratios['liquid']*100:.1f}% | Solid={ratios['solid']*100:.1f}% | "
              f"Avg Time Dilation (gamma)={avg_gamma:.3f}")

    # 2. Stage 2: Liquid Phase
    print("\n--- [Stage 2: Liquid Phase] Vortex Flow & Coherence Building ---")
    for step in range(5):
        # Dissipate energy and lower entropy
        V = V * 0.6
        output = engine(X, V, entropy=0.8)
        X, V = output["X"], output["V"]
        ratios = output["phase_ratios"]
        avg_gamma = float(torch.mean(output["Gammas"]).item())
        print(f"  Liquid Step {step+1}: Gas={ratios['gas']*100:.1f}% | "
              f"Liquid={ratios['liquid']*100:.1f}% | Solid={ratios['solid']*100:.1f}% | "
              f"Avg Time Dilation (gamma)={avg_gamma:.3f}")

    # 3. Stage 3: Solid Phase & Macro-Node Contraction
    print("\n--- [Stage 3: Solid Phase] Phase-Lock & Macro-Node Encapsulation ---")
    for step in range(5):
        V = V * 0.1
        output = engine(X, V, entropy=0.05)
        X, V = output["X"], output["V"]
        ratios = output["phase_ratios"]
        avg_gamma = float(torch.mean(output["Gammas"]).item())
        print(f"  Solid Step {step+1}: Gas={ratios['gas']*100:.1f}% | "
              f"Liquid={ratios['liquid']*100:.1f}% | Solid={ratios['solid']*100:.1f}% | "
              f"Avg Time Dilation (gamma)={avg_gamma:.3f}")

    X_macro, V_macro, cluster_map = engine.contract_macro_nodes(X, V, output["A"])
    print(f"  [*] Encapsulated {num_nodes} nodes into {X_macro.size(0)} Macro-Nodes.")

    # 4. Stage 4: Topological Shear Fracture
    print("\n--- [Stage 4: Topological Shear] Fracture Command & Recycling ---")
    impact_pt = X[0].clone()
    impact_v = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float32)
    X, V, fractured_mask = engine.apply_topological_shear(X, V, impact_pt, impact_v, radius=1.0)

    # Step again after fracture
    output = engine(X, V, entropy=2.0)
    ratios = output["phase_ratios"]
    print(f"  Post-Shear Step: Gas={ratios['gas']*100:.1f}% | "
          f"Liquid={ratios['liquid']*100:.1f}% | Solid={ratios['solid']*100:.1f}%")
    print(f"  [*] Fractured {int(fractured_mask.sum().item())} nodes back into active liquid/gas phases.")

    print("\n=========================================================================")
    print(" Simulation Completed Successfully!")
    print("=========================================================================")


if __name__ == "__main__":
    run_cognitive_crystallizer_demo()
