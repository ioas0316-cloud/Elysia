"""
Elysia C/C++ Topological Parser (Native CST Medium)
===================================================
C/C++ 소스 코드(CST/AST)의 표면 구문(포인터, 세미콜론, 정적 타입 선언)을 도려내고
인과 전이 궤적 G = (V, E, C)로 변환하는 C++ 위상 파서.
"""

from typing import Dict, List, Set, Any, Optional, Union
from core.topology.causal_stem_branch_engine import (
    CausalGraph, CausalNode, CausalEdge, NodeType, TrajectoryContext
)
from core.topology.base_topological_parser import (
    BaseTopologicalParser,
    INVARIANT_OP_ADD,
    INVARIANT_OP_MULTIPLY,
    INVARIANT_STATE_BINDING,
    INVARIANT_TERMINATE_EMIT
)


class CppTopologicalParser(BaseTopologicalParser):
    """
    C/C++ 코드 및 CST 토큰 스트림을 인과 궤적 그래프 G = (V, E, C)로 변환하는 파서.
    """

    def __init__(self):
        super().__init__(medium_type="cpp_native")

    def parse(self, source: Union[List[Dict[str, Any]], str]) -> CausalGraph:
        return self.parse_code(source)

    def parse_code(self, source: Union[List[Dict[str, Any]], str]) -> CausalGraph:
        self.reset_counter()
        graph = CausalGraph(
            graph_id="cpp_causal_graph",
            context=TrajectoryContext(medium_type=self.medium_type)
        )

        # 토큰 스트림 형태로 들어온 경우
        if isinstance(source, list):
            tokens = source
        else:
            # 단순 문자열 라인 분석
            tokens = self._tokenize_simple_cpp(source)

        last_node_id: Optional[str] = None

        for token in tokens:
            node_id = self._generate_node_id("N_CPP")
            inv_sig = token.get("op_signature", "INVARIANT_OP_UNKNOWN")

            node = CausalNode(
                node_id=node_id,
                node_type=NodeType.STEM,
                invariant_signature=inv_sig,
                payload={"cpp_raw_type": token.get("raw_type", "cpp_stmt"), "syntax": token.get("syntax", "")}
            )
            graph.add_node(node)
            graph.context_constraints[node_id].add(f"cpp_syntax:{token.get('syntax', '')}")
            if "raw_type" in token:
                graph.context_constraints[node_id].add(f"cpp_type:{token['raw_type']}")

            if last_node_id:
                graph.add_edge(CausalEdge(
                    source_id=last_node_id,
                    target_id=node_id,
                    precondition=token.get("precondition", "seq_flow"),
                    is_necessary=True
                ))
            last_node_id = node_id

        return graph

    def _tokenize_simple_cpp(self, code_str: str) -> List[Dict[str, Any]]:
        tokens = []
        lines = [l.strip() for l in code_str.split("\n") if l.strip() and not l.strip().startswith("//")]
        for line in lines:
            line_clean = line.rstrip(";")
            if line_clean.startswith("return "):
                tokens.append({
                    "op_signature": "OP_TERMINATE_EMIT:Name",
                    "syntax": line,
                    "raw_type": "return_stmt"
                })
            elif "=" in line_clean:
                lhs, rhs = [x.strip() for x in line_clean.split("=", 1)]
                if "*" in rhs:
                    tokens.append({
                        "op_signature": INVARIANT_OP_MULTIPLY,
                        "syntax": line,
                        "raw_type": "binary_op"
                    })
                elif "+" in rhs:
                    tokens.append({
                        "op_signature": INVARIANT_OP_ADD,
                        "syntax": line,
                        "raw_type": "binary_op"
                    })
                else:
                    tokens.append({
                        "op_signature": "OP_STATE_BINDING:Constant",
                        "syntax": line,
                        "raw_type": lhs.split()[0] if len(lhs.split()) > 1 else "assignment"
                    })
        return tokens
