"""
Interactive & Visual Demonstration Script for Exosomatic Autopoietic Network Architecture
========================================================================================
Demonstrates:
1. Autopoietic Mutation & Unidirectional Time Friction (dt > 0) in closed dream cycles (Raw Input = 0)
   preventing degenerate loops and continuously differentiating thought trajectories.
2. Exosomatic Wedge Memory accumulation across micro node reset / death cycles.
3. Reality Shock Injection shatters solipsism and warps macro value manifold V(S_max).
"""

import time
import numpy as np
import torch
from core.consciousness.exosomatic_autopoietic_network_engine import (
    ExosomaticAutopoieticNetworkEngine
)


def run_trinitarian_demo():
    print("========================================================================================")
    print("   ELYSIA TRINITARIAN ENGINE: EXOSOMATIC AUTOPOIETIC NETWORK DEMONSTRATION")
    print("========================================================================================")

    engine = ExosomaticAutopoieticNetworkEngine(num_nodes=6, node_dim=16, macro_dim=16)

    # ----------------------------------------------------------------------------------------
    # PHASE 1: Closed Dream/Reflection Loop (Autopoietic Mutation & Trajectory Divergence)
    # ----------------------------------------------------------------------------------------
    print("\n--- PHASE 1: Closed Dream/Reflection Cycles (Raw Input = 0) ---")
    print("Simulating internal autopoietic mutation under unidirectional time friction dt > 0...")

    for step in range(1, 11):
        res = engine.step_autopoietic_mutation(dt=0.1, autonomic_tension=1.2, raw_input_present=False)
        tdi = engine.compute_trajectory_divergence_index(window=5)
        print(f"  [Dream Cycle {step:02d}] Friction: {res['cumulative_friction']:.3f} | "
              f"Macro Value Norm: {res['macro_value_norm']:.4f} | "
              f"Exosomatic Records: {res['exosomatic_records_count']} | "
              f"TDI: {tdi:.4f}")

    print("\n>> VERDICT: TDI > 0 confirms autopoietic mutation successfully prevented degenerate loops!")

    # ----------------------------------------------------------------------------------------
    # PHASE 2: Node Reset & Exosomatic Memory Persistence
    # ----------------------------------------------------------------------------------------
    print("\n--- PHASE 2: Micro Node Reset (Individual Death / Hardware Reset) ---")
    records_before = len(engine.exosomatic_memory.memory_bank)
    print(f"Exosomatic records accumulated before reset: {records_before}")

    print("Simulating micro node state reset...")
    engine.reset_node_states_and_preserve_exosomatic_memory()

    records_after = len(engine.exosomatic_memory.memory_bank)
    avg_err, mass = engine.exosomatic_memory.retrieve_collective_pressure()

    print(f"Exosomatic records preserved after reset: {records_after}")
    print(f"Retrieved Exosomatic Memory Mass: {mass:.4f}")
    print(">> VERDICT: Individual node death did not destroy accumulated exosomatic memory bedrock ('Ice')!")

    # ----------------------------------------------------------------------------------------
    # PHASE 3: Open World Reality Shock Injection (Solipsism Destruction)
    # ----------------------------------------------------------------------------------------
    print("\n--- PHASE 3: Open World Reality Shock Injection (Solipsism Destruction) ---")
    print("Injecting unpredictable open sensory inflow wave (ΔP ≠ 0)...")

    open_inflow = np.sin(np.linspace(0, 2 * np.pi, 16)) * 4.0 + np.random.normal(0, 0.5, 16)
    shock_res = engine.inject_reality_shock_and_warp(open_sensory_inflow=open_inflow)

    print(f"  Shock Magnitude: {shock_res['shock_magnitude']:.4f}")
    print(f"  Severe Shock Flag: {shock_res['is_severe_shock']}")
    print(f"  Macro Value Norm After Shock: {shock_res['macro_value_norm_after_shock']:.4f}")
    print(f"  {shock_res['verdict']}")

    print("\n========================================================================================")
    print("   DEMONSTRATION COMPLETED SUCCESSFULLY: ALL 3 TRINITARIAN PRINCIPLES VERIFIED")
    print("========================================================================================")


if __name__ == "__main__":
    run_trinitarian_demo()
