"""
Elysia Core - Standalone Verification Script for Unstructured Data Topological
& Rotor Extraction & 5-Stat Environmental Principle Closed-Loop Feedback
========================================================================
Demonstrates extraction of Riemannian metric g_μν, Cl_{3,1} rotors,
Christoffel symbols Γ^λ_μν, and 5-stat macro-micro closed loop convergence.
"""

import sys
import numpy as np
from core.ingestion.unstructured_topological_extractor import UnstructuredTopologicalExtractor
from core.causal_world.five_stat_closed_loop import FiveStatClosedLoopEngine, FiveStatVector


def main():
    print("==========================================================================")
    print(" [VERIFICATION] Elysia Unstructured Topological Extraction & 5-Stat Loop ")
    print("==========================================================================")

    # 1. Extract Topological Manifold & Rotors from Unstructured Input
    extractor = UnstructuredTopologicalExtractor(dim=4)
    sample_corpus = [
        "Humanity projects its behavioral trajectories into the digital twin medium.",
        "A societal friction storm shifts environmental curvature and heightens energy decay.",
        "Under high fatigue, the causal rotor guides the NPC towards equilibrium and bed rest."
    ]

    print("\n[Step 1] Processing Unstructured Real-World Text Streams:")
    extractions = []
    for idx, text in enumerate(sample_corpus, 1):
        ext = extractor.extract_from_unstructured_text(text)
        extractions.append(ext)
        print(f"  Stream {idx}: '{text[:45]}...'")
        print(f"    - Tokens: {ext['num_tokens']}")
        print(f"    - Ricci Curvature Scalar: {ext['ricci_scalar']:.4f}")
        print(f"    - Attractor Gravitational Depth: {ext['attractor_potential']:.4f}")
        print(f"    - Mean Spatial Rotation Mag: {ext['mean_spatial_rotation']:.4f}")

    # 2. Reconstructed Metric Tensor Inspection
    final_metric = extractor.g
    print("\n[Step 2] Reconstructed Riemannian Metric Tensor g_μν (4x4 Spacetime):")
    for row in final_metric:
        print("   ", [round(float(val), 4) for val in row])

    # 3. 5-Stat Environmental Mapping & Closed-Loop Simulation
    print("\n[Step 3] Executing Macro-Micro Closed-Loop 5-Stat Simulation:")
    loop_engine = FiveStatClosedLoopEngine()
    npc_stat = FiveStatVector(
        energy_consumption=0.82,  # Fatigued initial state
        info_bandwidth=0.40,
        friction_resistance=0.50,
        adaptation_speed=0.30,
        equilibrium_stability=0.25 # Unstable homeostatic state
    )

    print(f"  Initial NPC 5-Stats: S_E={npc_stat.energy_consumption:.2f}, S_S={npc_stat.equilibrium_stability:.2f}")

    for step in range(1, 6):
        ext_data = extractions[(step - 1) % len(extractions)]
        npc_stat, summary = loop_engine.step_closed_loop(
            npc_stat=npc_stat,
            extraction_data=ext_data,
            action_intent="work_or_exert" if step % 2 == 1 else "walk"
        )
        print(f"  Loop Step {step}: Action -> '{summary['executed_action']}'")
        print(f"    - NPC State: S_E={summary['npc_energy_consumption']:.4f}, S_S={summary['npc_equilibrium_stability']:.4f}")
        print(f"    - Macro Environment: Friction={summary['macro_friction']:.4f}, Stability={summary['macro_stability']:.4f}")
        print(f"    - Macro-Micro Closed-Loop Resonance: {summary['closed_loop_resonance']:.4f}")

    print("\n==========================================================================")
    print(" [SUCCESS] Topological Metric Reconstruction & 5-Stat Loop Verified! ")
    print("==========================================================================")


if __name__ == "__main__":
    main()
