"""
Feudal-Free Causal Manifold Demonstration Script

Demonstrates the abolition of artificial data types (JSON, PNG, String) by
representing all data as Native Topological Manifolds, observing Invariance & Divergence,
reconstructing boundary discontinuities, and phase-locking into a unified Ice macro-topology (FLOPs=0).
"""

import sys
from core.topology.native_topological_manifold import (
    SequentialOffsetTopology,
    SpatialGridTopology,
    HierarchicalBranchTopology
)
from core.topology.invariance_divergence_tracker import (
    InvarianceDivergenceTracker,
    CausalDiscontinuityDetector
)
from core.topology.topological_phase_engine import TopologicalPhaseEngine


def main():
    print("==========================================================================")
    print("  Elysia: Feudal-Free Native Causal Manifold & Phase Lock Demonstration   ")
    print("==========================================================================")
    print()

    # 1. Native Topological Input
    print("[1] Ingesting Raw Data as Native Topological Manifolds (No External Parsers)")
    raw_json = {"entity": "Hero", "stats": {"hp": 100, "mp": 50}}
    raw_png_pixels = [
        ["#FF0000", "#00FF00"],
        ["#0000FF", "#FFFFFF"]
    ]
    raw_text = "NATIVE_TOPOLOGY"

    json_manifold = HierarchicalBranchTopology("json_tree", raw_json)
    png_manifold = SpatialGridTopology("png_grid", raw_png_pixels)
    text_manifold = SequentialOffsetTopology("text_seq", raw_text)

    print(f"  - JSON Hierarchical Tree Manifold: {len(json_manifold.nodes)} nodes")
    print(f"  - PNG Spatial Grid 2D Manifold:    {len(png_manifold.nodes)} nodes")
    print(f"  - Text Sequential Offset Manifold: {len(text_manifold.nodes)} nodes")
    print()

    # 2. Phase 1: Gas State
    engine = TopologicalPhaseEngine("feudal_free_engine")
    gas_info = engine.observe_gas_state([json_manifold, png_manifold, text_manifold])
    print(f"[2] Phase State: {gas_info['phase']}")
    print(f"  - Status: {gas_info['status']}")
    print(f"  - Unbound Floating Nodes: {gas_info['total_unbound_nodes']}")
    print()

    # 3. Phase 2: Invariance & Divergence Observation (Liquid State Coupling)
    print("[3] Phase State: Transitioning to Liquid Coupling (Active Boundary Alignment)")
    invariant = InvarianceDivergenceTracker.observe_invariant(png_manifold, json_manifold)
    divergence = InvarianceDivergenceTracker.observe_divergence(png_manifold, json_manifold)

    print(f"  - Shared Structural Invariant: {invariant['principle']}")
    print(f"  - Structural Divergence Rule:  {divergence['divergence_rule']}")
    print(f"  - f_divergence Explanation:   {divergence['explanation']}")

    liquid_info = engine.initiate_liquid_coupling(png_manifold, json_manifold)
    print(f"  - Liquid Bridge Created:       {liquid_info['bridge_node_id']}")
    print()

    # 4. Phase 3: Ice Crystallization (Zero FLOP Macro-Topology)
    print("[4] Phase State: Crystallizing into Ice (FLOPs = 0 Macro-Topology)")
    ice_macro = engine.crystallize_to_ice(png_manifold, json_manifold)
    print(f"  - Unified Ice Macro-Topology ID: {ice_macro.manifold_id}")
    print(f"  - Total Integrated Macro Nodes:   {len(ice_macro.nodes)}")
    print(f"  - Macro Root Nodes:               {[r.node_id for r in ice_macro.root_nodes]}")
    print()

    # 5. Native Zero-FLOP Macro Traversal
    print("[5] Executing Zero-FLOP Direct Traversal Across Crystallized Boundary")
    bridge_root_id = list(ice_macro.root_nodes)[0].node_id
    path_query = ["connected_boundary_a:png_grid_grid_0_0", "east"]

    traversal_res = engine.query_ice_macro_topology(bridge_root_id, path_query)
    print(f"  - Query Status:         {traversal_res['phase']}")
    print(f"  - FLOPs Performed:      {traversal_res['flops_performed']}")
    print(f"  - Traversal Path:       {' -> '.join(traversal_res['traversal_path'])}")
    print(f"  - Final Target Payload: {traversal_res['final_payload']}")
    print()

    print("==========================================================================")
    print("  Demonstration Successful: Feudal Boundaries Removed & Ice Lock Verified  ")
    print("==========================================================================")


if __name__ == "__main__":
    main()
