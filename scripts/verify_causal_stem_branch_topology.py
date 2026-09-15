"""
Verification Script: Homological Stem & Branch Topology Engine
==============================================================
파편적 누더기 인과 데이터가 위상 해부학 엔진을 거쳐
온전한 인과 지도로 환원되는 시뮬레이션 검증 데모.
"""

from core.topology.causal_stem_branch_engine import (
    CausalStemBranchEngine,
    CausalTrajectoryGraph,
    CausalNode,
    CausalEdge,
    TrajectoryContext
)


def run_verification():
    print("=" * 70)
    print(" [Elysia Core] Homological Stem & Branch Topology Verification")
    print("=" * 70)

    # 1. Trajectory Graph A: Software System Code Execution (Python Medium)
    graph_A = CausalTrajectoryGraph(
        graph_id="software_system_ast",
        context=TrajectoryContext(medium_type="python_ast")
    )
    graph_A.add_node(CausalNode(id="ast_in", role="input", abstract_operation="receive_data_stream"))
    graph_A.add_node(CausalNode(id="python_gc", role="state", abstract_operation="memory_garbage_collection")) # Branch
    graph_A.add_node(CausalNode(id="ast_transform", role="transform", abstract_operation="causal_state_transition"))
    graph_A.add_node(CausalNode(id="ast_out", role="output", abstract_operation="emit_action_state"))

    graph_A.add_edge(CausalEdge(source_id="ast_in", target_id="python_gc", mechanism_type="medium_runtime_hook"))
    graph_A.add_edge(CausalEdge(source_id="ast_in", target_id="ast_transform", mechanism_type="direct_flow"))
    graph_A.add_edge(CausalEdge(source_id="ast_transform", target_id="ast_out", mechanism_type="direct_flow"))

    # 2. Trajectory Graph B: Physical Neural/Biological Circuit (Physical Medium)
    graph_B = CausalTrajectoryGraph(
        graph_id="biological_circuit",
        context=TrajectoryContext(medium_type="neural_membrane")
    )
    graph_B.add_node(CausalNode(id="synapse_in", role="input", abstract_operation="receive_data_stream"))
    graph_B.add_node(CausalNode(id="ion_channel", role="transform", abstract_operation="causal_state_transition"))
    graph_B.add_node(CausalNode(id="heat_dissipation", role="state", abstract_operation="metabolic_heat_loss")) # Branch
    graph_B.add_node(CausalNode(id="action_potential", role="output", abstract_operation="emit_action_state"))

    graph_B.add_edge(CausalEdge(source_id="synapse_in", target_id="ion_channel", mechanism_type="direct_flow"))
    graph_B.add_edge(CausalEdge(source_id="ion_channel", target_id="action_potential", mechanism_type="direct_flow"))
    graph_B.add_edge(CausalEdge(source_id="action_potential", target_id="heat_dissipation", mechanism_type="medium_runtime_hook"))

    print(f"\n[Step 1] Graph Decomposition:")
    print(f" - Graph A ({graph_A.graph_id}, Medium: {graph_A.context.medium_type}): {len(graph_A.nodes)} nodes, {len(graph_A.edges)} edges")
    print(f" - Graph B ({graph_B.graph_id}, Medium: {graph_B.context.medium_type}): {len(graph_B.nodes)} nodes, {len(graph_B.edges)} edges")

    # 3. Parse Stem & Branches
    engine = CausalStemBranchEngine()
    result = engine.parse_stem_and_branches(graph_A, graph_B)

    # 4. Display Results
    print(f"\n[Step 2] Homological Stem Extraction (1:1 Isomorphic Invariant):")
    print(f" - Stem ID: {result.stem.stem_id}")
    print(f" - Common Node Count: {len(result.stem.common_subgraph_A_node_ids)}")
    print(f" - 1:1 Isomorphic Mapping f(A) -> B:")
    for src_a, tgt_b in result.stem.node_mapping_A_to_B.items():
        print(f"    * Node '{src_a}' ({graph_A.nodes[src_a].abstract_operation}) <===> Node '{tgt_b}' ({graph_B.nodes[tgt_b].abstract_operation})")

    print(f"\n[Step 3] Disparate Branch Isolation (Medium-Specific Variance):")
    print(f" - Graph A Branches ({len(result.branches_A)} branch isolated):")
    for br in result.branches_A:
        print(f"    * Branch Nodes: {br.branch_node_ids}, Attached to Stem Node: '{br.attached_stem_node_id}'")
    print(f" - Graph B Branches ({len(result.branches_B)} branch isolated):")
    for br in result.branches_B:
        print(f"    * Branch Nodes: {br.branch_node_ids}, Attached to Stem Node: '{br.attached_stem_node_id}'")

    print(f"\n[Step 4] Trajectory Continuity Verification:")
    print(f" - Is Continuous Trajectory? : {result.is_continuous}")
    if not result.is_continuous:
        print(f" - Discontinuity Reasons: {result.discontinuity_reasons}")
    else:
        print(" - Verification Success: Input-to-Output Causal Flow is Deterministically Preserved Across Mediums!")

    print("\n" + "=" * 70)
    print(" Homological Stem & Branch Topology Verification Complete Successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
