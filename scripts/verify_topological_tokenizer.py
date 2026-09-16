import sys
import os

sys.path.append(os.path.abspath("."))

from core.topology.topological_tokenizer import TopologicalTokenizer, NodeType

def run_verification():
    tokenizer = TopologicalTokenizer()

    test_text = "한 ㅋㅋㅋ"
    print(f"=== 위상 토크나이저 검증 실행 ===")
    print(f"입력 텍스트: '{test_text}'\n")

    causal_graph = tokenizer.tokenize_text(test_text)

    print("--- [1. 생성된 노드(Nodes) 목록] ---")
    for node_id, node in causal_graph.nodes.items():
        ctx = causal_graph.context_constraints.get(node_id, set())
        print(f"ID: {node_id:<14} | Type: {node.node_type.value:<6} | Sig: {node.invariant_signature:<30} | Context: {sorted(list(ctx))}")

    print("\n--- [2. 인과 결합 엣지(Edges) 목록] ---")
    for edge in causal_graph.edges:
        nec = "필연(Necessary)" if edge.is_necessary else "우연(Optional)"
        print(f"Edge: {edge.source_id:<14} ===[{edge.precondition}]===> {edge.target_id:<14} ({nec})")

    print("\n--- [3. '한' 음절의 자모 원자 해부 검증] ---")
    han_nodes = [n for n in causal_graph.nodes.values() if "INVARIANT_SYLLABLE:한" in n.invariant_signature]
    if han_nodes:
        han_node = han_nodes[0]
        print(f"✓ 음절 '한' STEM 노드 확인: {han_node.node_id} ({han_node.invariant_signature})")

        inbound_edges = [e for e in causal_graph.edges if e.target_id == han_node.node_id]
        for e in inbound_edges:
            parent_node = causal_graph.nodes[e.source_id]
            print(f"  └─ 합성 전단계 원자 노드: {parent_node.node_id} ({parent_node.invariant_signature})")

    print("\n--- [4. 'ㅋㅋㅋ' 비정규 표현(Branch) 격리 검증] ---")
    branch_nodes = [n for n in causal_graph.nodes.values() if n.node_type == NodeType.BRANCH]
    for b_node in branch_nodes:
        ctx = causal_graph.context_constraints.get(b_node.node_id, set())
        print(f"✓ BRANCH 노드 격리 확인: {b_node.node_id} | Sig: {b_node.invariant_signature} | 제약조건: {sorted(list(ctx))}")

if __name__ == "__main__":
    run_verification()
