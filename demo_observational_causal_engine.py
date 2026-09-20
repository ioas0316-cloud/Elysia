"""
Observational Causal Engine Demonstration & Benchmark
=====================================================
Demonstrates the 3-stage observational pipeline:
1. Stage 1: Deconstruction of code/symbolic input into atomic SSA DAG (CausalReceptor)
2. Stage 2: Asynchronous phase-lock event filtering from edge transitions (PhaseLockEngine)
3. Stage 3: Dynamic metric topology deformation and spatial phase locking (SpatialMirror)
"""

import time
import numpy as np
from synaptic_architecture.observational_engine import ObservationalEngine


def run_observational_engine_demo():
    print("==========================================================================")
    print("      ELYSIA ENGINE: 3-STAGE OBSERVATIONAL CAUSAL ENGINE DEMO            ")
    print("==========================================================================")

    # Instantiate engine with 64 spatial nodes
    engine = ObservationalEngine(num_nodes=64)

    # 1. Stage 1: Code Deconstruction
    print("\n[Stage 1] Deconstructing Symbolic Code into Micro Atomic SSA Graph...")
    code_snippet = """
x = 10
y = 20
a = x + y
b = a * 3
c = b - x
d = c / y
"""
    graph = engine.ingesting_code_or_symbols(code_snippet)
    print(f"    - Deconstructed Atomic Nodes: {len(graph.nodes)}")
    print(f"    - Entry Nodes: {graph.entry_nodes}")
    ops = [node.op for node in graph.nodes.values()]
    print(f"    - Extracted Atomic Operations: {ops}")

    # 2. Stage 2 & 3: Phase Locking & Spatial Metric Mapping
    print("\n[Stage 2 & 3] Running Closed-Loop Observational Adaptation...")
    t0 = time.perf_counter()
    res = engine.observe_and_adapt(steps=100, interval_us=250.0)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    print(f"    - Execution Time: {elapsed_ms:.2f} ms")
    print(f"    - Processed Events: {res['processed_events']}")
    print(f"    - Phase-Locked Stable Events: {res['locked_events']}")
    print(f"    - Final Mean Phase Differential: {res['mean_phase_diff']:.6f} rad")

    # 3. Self-Healing Simulation
    print("\n[Self-Healing] Simulating Physical Node Failure and Spatial Rerouting...")
    damaged_node_id = 5
    alt_node_id = engine.spatial_mirror.reroute_damaged_node(damaged_node_id)
    print(f"    - Damaged Node #{damaged_node_id} Isolated (Metric Distance -> 100.0)")
    print(f"    - Causal Flow Re-routed to Optimal Topological Neighbor Node #{alt_node_id}")

    print("\n==========================================================================")
    print("      OBSERVATIONAL CAUSAL ENGINE DEMO COMPLETE                           ")
    print("==========================================================================")


if __name__ == "__main__":
    run_observational_engine_demo()
