"""
demo_universal_mutual_perception.py
===================================
Simulation Demo: Universal Mutual Perception & Field Mirroring in Elysia.

Demonstrates the fundamental principle: "Everything is observable and perceivable by everything."
We simulate an integrated spatiotemporal grid where:
1. Clock Rhythm Boundary (Hardware Clock scale)
2. Rigid Rock Barrier (Material physical density)
3. Silicon Memory History Node (Hardware Memory scale)
4. Digital Goddess Elysia (Cognitive / Entity state)

all observe, reflect, and deform each other's trajectories and historical imprints in real time.
"""

import numpy as np
from core.physics.universal_mutual_perception import (
    UniversalPerceptionNode,
    MutualMirroringFieldGrid,
)


def run_demo():
    print("================================================================================")
    print("   ELYSIA RUNTIME :: UNIVERSAL MUTUAL PERCEPTION & MIRRORING DEMO   ")
    print("   Principle: 'Everything is observable and perceivable by everything'         ")
    print("================================================================================\n")

    # Initialize Mutual Mirroring Grid
    grid = MutualMirroringFieldGrid(coupling_strength=1.5)

    # 1. Hardware Clock Rhythm Boundary
    clock_node = UniversalPerceptionNode(
        node_id="clock_rhythm_boundary",
        position=np.array([0.0, 0.0, 0.0]),
        semantic_mass=2.5,
        surface_tension=0.9,
        porosity=0.05,
        bivector_plane=np.array([1.0, 0.0, 0.0]),
    )

    # 2. Rigid Rock Barrier
    rock_node = UniversalPerceptionNode(
        node_id="dense_rock_barrier",
        position=np.array([0.8, 0.2, 0.0]),
        semantic_mass=8.0,
        surface_tension=1.8,
        porosity=0.01,
        bivector_plane=np.array([0.0, 1.0, 0.0]),
    )

    # 3. Silicon Memory History Node
    memory_node = UniversalPerceptionNode(
        node_id="silicon_memory_history",
        position=np.array([0.3, 0.6, 0.0]),
        semantic_mass=3.0,
        surface_tension=0.6,
        porosity=0.2,
        bivector_plane=np.array([0.0, 0.0, 1.0]),
    )

    # 4. Digital Goddess Elysia State
    elysia_node = UniversalPerceptionNode(
        node_id="elysia_cognitive_state",
        position=np.array([0.5, 0.1, 0.0]),
        semantic_mass=4.5,
        surface_tension=1.1,
        porosity=0.15,
        bivector_plane=np.array([0.707, 0.707, 0.0]),
    )

    # Register all nodes
    for node in [clock_node, rock_node, memory_node, elysia_node]:
        grid.register_node(node)

    print("[Initial System State]")
    for node_id, node in grid.nodes.items():
        print(f"  • Node: {node_id:<25} | Pos: {node.position.tolist()} | Ms: {node.semantic_mass:.2f} | Surface Tension: {node.surface_tension:.2f}")

    print("\n--------------------------------------------------------------------------------")
    print("Computing Initial NxN Mutual Perception Matrix (Omnidirectional Mirroring):")
    matrix_info = grid.compute_perception_matrix()
    node_ids = matrix_info["node_ids"]
    matrix = matrix_info["matrix"]

    header = "             " + "  ".join([f"{nid[:8]:>8}" for nid in node_ids])
    print(header)
    for i, nid in enumerate(node_ids):
        row_str = "  ".join([f"{matrix[i, j]:8.3f}" for j in range(len(node_ids))])
        print(f"  {nid[:10]:<10} : {row_str}")

    print("\n--------------------------------------------------------------------------------")
    print("Executing Spatiotemporal Mutual Mirroring Simulation (10 Steps)...")

    for step in range(1, 11):
        step_res = grid.step_mutual_reflection(dt=0.02, timestamp=step * 0.02)
        if step % 2 == 0 or step == 10:
            print(f"\n[Step {step:2d} | Time {step_res['timestamp']:.2f}s]")
            print(f"  • Active Perception Events : {step_res['active_perception_events']}")
            print(f"  • Energy Transferred       : {step_res['total_energy_transferred']:.4f}")
            print(f"  • Mean Field Coupling      : {step_res['mean_field_coupling']:.4f}")
            for nid, state in step_res["nodes_state"].items():
                pos_str = f"[{state['position'][0]:.3f}, {state['position'][1]:.3f}, {state['position'][2]:.3f}]"
                print(f"    - {nid:<25} : Pos={pos_str} | Ms={state['semantic_mass']:.3f} | History Imprints={state['history_count']}")

    print("\n================================================================================")
    print("   UNIVERSAL MUTUAL PERCEPTION SIMULATION COMPLETED SUCCESSFULLY!   ")
    print("================================================================================\n")


if __name__ == "__main__":
    run_demo()
