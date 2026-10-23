"""
Elysia SymPy Topological Parser (Mathematics/Logic Medium)
==========================================================
SymPy 수식 및 기호 논리 표현식을 CausalGraph(V, E, C) 구조로 해부하는 파서.
수학 기호 표기법을 탈색하고 인과 전이 위상 서명으로 사상한다.
"""

import sympy as sp
from typing import Dict, List, Set, Any, Optional
from core.topology.causal_stem_branch_engine import (
    CausalGraph, CausalNode, CausalEdge, NodeType, TrajectoryContext
)
from core.topology.base_topological_parser import (
    BaseTopologicalParser,
    INVARIANT_OP_ADD,
    INVARIANT_OP_MULTIPLY,
    INVARIANT_OP_EQUALS,
    INVARIANT_VAR_BINDING,
    INVARIANT_CONST_VALUE
)


class SymPyTopologicalParser(BaseTopologicalParser):
    """
    SymPy 수식 및 논리 표현식을 CausalGraph(V, E, C) 구조로 해부하는 파서.
    """

    def __init__(self):
        super().__init__(medium_type="sympy_expression")

    def parse(self, expr: sp.Expr) -> CausalGraph:
        return self.parse_expr(expr)

    def parse_expr(self, expr: sp.Expr) -> CausalGraph:
        self.reset_counter()
        graph = CausalGraph(
            graph_id="sympy_causal_graph",
            context=TrajectoryContext(medium_type=self.medium_type)
        )
        self._traverse(expr, graph, incoming_ids=[], precondition="math_flow")
        return graph

    def _traverse(
        self, 
        node: Any, 
        graph: CausalGraph, 
        incoming_ids: List[str], 
        precondition: str
    ) -> List[str]:
        node_id = self._generate_node_id("N_MATH")
        
        # SymPy 연산자 타입을 범용 인과 서명으로 표준화
        if isinstance(node, sp.Add):
            inv_sig = INVARIANT_OP_ADD
        elif isinstance(node, sp.Mul):
            inv_sig = INVARIANT_OP_MULTIPLY
        elif isinstance(node, sp.Equality):
            inv_sig = INVARIANT_OP_EQUALS
        elif isinstance(node, sp.Symbol):
            inv_sig = INVARIANT_VAR_BINDING
        elif isinstance(node, (sp.Integer, sp.Float, sp.Number)):
            inv_sig = INVARIANT_CONST_VALUE
        else:
            inv_sig = f"INVARIANT_OP_{node.__class__.__name__.upper()}"

        causal_node = CausalNode(
            node_id=node_id,
            node_type=NodeType.STEM,
            invariant_signature=inv_sig,
            payload={"sympy_class": node.__class__.__name__, "repr": str(node)}
        )
        graph.add_node(causal_node)
        graph.context_constraints[node_id].add(f"sympy_type:{node.__class__.__name__}")

        for inc_id in incoming_ids:
            graph.add_edge(CausalEdge(
                source_id=inc_id,
                target_id=node_id,
                precondition=precondition,
                is_necessary=True
            ))

        # 하위 서브 트리가 있는 경우 재귀적 위상 연결
        current_exits = [node_id]
        if hasattr(node, "args") and node.args:
            child_exits = []
            for arg in node.args:
                child_exits.extend(self._traverse(arg, graph, current_exits, precondition="arg_dependency"))
            return child_exits

        return current_exits
