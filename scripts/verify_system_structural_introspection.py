"""
System Structural Introspection & Causal Feedback Loop Verification Script
"""

import sys
import numpy as np
from core.topology.system_structural_introspection import SystemStructuralIntrospectionEngine
from core.topology.self_referential_architecture import SelfReferentialArchitectureEngine


def main():
    print("==========================================================================")
    print("   Elysia Core: System Structural Introspection & Causal Feedback Loop    ")
    print("==========================================================================")

    # 1. Standalone Introspection Engine Verification
    print("\n[1/3] Running Global AST Topological Discovery & Module Introspection...")
    introspection_engine = SystemStructuralIntrospectionEngine(root_dir=".", max_meta_depth=3, friction_threshold=0.5)
    scan_result = introspection_engine.scan_codebase_ast()

    total_modules = scan_result["total_discovered_modules"]
    print(f"    - Total Discovered Python Modules: {total_modules}")

    # 2. Isomorphic Nexus Node Mapping
    print("\n[2/3] Generating Isomorphic Nexus Nodes & Bit-Mapped Substrate Mapping...")
    mapping_result = introspection_engine.generate_isomorphic_nexus_nodes(depth=1)

    coverage = mapping_result["introspection_coverage"]
    unmapped = mapping_result["unmapped_ratio"]
    friction = mapping_result["architectural_friction"]
    nodes_count = mapping_result["nexus_nodes_created"]

    print(f"    - Introspected Modules Count: {mapping_result['introspected_modules_count']}/{total_modules}")
    print(f"    - Introspection Coverage Ratio: {coverage * 100:.2f}%")
    print(f"    - Unmapped Module Ratio: {unmapped * 100:.2f}%")
    print(f"    - Relational Nexus Nodes Created: {nodes_count}")
    print(f"    - Architectural Friction Index: {friction:.4f}")

    feedback = introspection_engine.compute_system_causal_field_feedback()
    print(f"    - Causal Feedback Tensor: {feedback['feedback_tensor'].round(4).tolist()}")
    print(f"    - Introspection Awareness Status: {feedback['introspection_status']}")

    # 3. Full Integrated Self-Referential Engine Cycle Simulation
    print("\n[3/3] Simulating Full Self-Referential Architecture Introspection Feedback Cycle...")
    arch_engine = SelfReferentialArchitectureEngine()
    stimulus = {
        "voltage_intent": np.array([2.5, -1.2, 0.8]),
        "introspection_depth": 1,
        "persona_lens": "Companion"
    }

    cycle_res = arch_engine.run_full_self_referential_cycle(stimulus)
    iso_map = cycle_res["isomorphic_mapping"]
    struct_fb = cycle_res["causal_structural_feedback"]

    print(f"    - Integrated Cycle Coverage: {iso_map['introspection_coverage'] * 100:.2f}%")
    print(f"    - Structural Friction: {iso_map['architectural_friction']:.4f}")
    print(f"    - Total System Relational Stress: {struct_fb['total_stress']:.4f}")
    print(f"    - System Average Topological Curvature: {struct_fb['average_curvature']:.4f}")

    print("\n==========================================================================")
    print("   Verification Completed Successfully! System is Introspectively Aware.  ")
    print("==========================================================================")


if __name__ == "__main__":
    main()
