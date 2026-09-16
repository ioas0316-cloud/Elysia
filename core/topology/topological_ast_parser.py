"""
Elysia Topological AST Parser (Python Medium)
==============================================
Python 소스 코드를 읽어 구문적 껍데기(변수명 표기, 라인번호, 언어 구문)를 탈색하고
범용 인과 서명(Universal Invariant Signature) 기반의 CausalGraph G = (V, E, C)로 사상하는 파서.
"""

import ast
from typing import Dict, List, Set, Any, Optional
from core.topology.causal_stem_branch_engine import (
    CausalGraph, CausalNode, CausalEdge, NodeType, TrajectoryContext
)
from core.topology.base_topological_parser import (
    BaseTopologicalParser,
    INVARIANT_OP_ADD,
    INVARIANT_OP_SUBTRACT,
    INVARIANT_OP_MULTIPLY,
    INVARIANT_OP_DIVIDE,
    INVARIANT_STATE_BINDING,
    INVARIANT_CONST_VALUE,
    INVARIANT_VAR_BINDING,
    INVARIANT_TERMINATE_EMIT
)


class TopologicalASTParser(BaseTopologicalParser):
    """
    Python AST를 인과 궤적 그래프 G = (V, E, C)로 변환하는 위상 파서.
    """

    def __init__(self):
        super().__init__(medium_type="python_ast")

    def parse(self, source: str) -> CausalGraph:
        return self.parse_code(source)

    def parse_code(self, source: str) -> CausalGraph:
        self.reset_counter()
        graph = CausalGraph(
            graph_id="python_ast_graph",
            context=TrajectoryContext(medium_type=self.medium_type)
        )
        parsed_tree = ast.parse(source.strip())
        
        last_seq_node_id: Optional[str] = None

        for stmt in parsed_tree.body:
            node_ids = self._parse_statement(stmt, graph, last_seq_node_id)
            if node_ids:
                last_seq_node_id = node_ids[-1]

        return graph

    def _parse_statement(
        self,
        stmt: ast.stmt,
        graph: CausalGraph,
        incoming_seq_id: Optional[str]
    ) -> List[str]:
        created_node_ids: List[str] = []

        if isinstance(stmt, ast.Assign):
            # 변수 할당: 우변 표현식 파싱 후 상태 바인딩 노드 생성
            val_node_ids = self._parse_expr(stmt.value, graph)
            created_node_ids.extend(val_node_ids)

            bind_node_id = self._generate_node_id("N_PY_BIND")
            # 우변이 단순 상수인 경우와 연산인 경우 구분
            if isinstance(stmt.value, ast.Constant):
                inv_sig = "OP_STATE_BINDING:Constant"
            else:
                inv_sig = INVARIANT_STATE_BINDING

            node = CausalNode(
                node_id=bind_node_id,
                node_type=NodeType.STEM,
                invariant_signature=inv_sig,
                payload={"targets": [t.id for t in stmt.targets if isinstance(t, ast.Name)]}
            )
            graph.add_node(node)
            graph.context_constraints[bind_node_id].add(f"py_lineno:{stmt.lineno}")
            graph.context_constraints[bind_node_id].add("py_syntax:ast.Assign")

            for v_id in val_node_ids:
                graph.add_edge(CausalEdge(
                    source_id=v_id,
                    target_id=bind_node_id,
                    precondition="value_flow",
                    is_necessary=True
                ))

            if incoming_seq_id:
                graph.add_edge(CausalEdge(
                    source_id=incoming_seq_id,
                    target_id=bind_node_id,
                    precondition="seq_flow",
                    is_necessary=True
                ))

            created_node_ids.append(bind_node_id)

        elif isinstance(stmt, ast.Return):
            # 반환문: 종결 및 방출 노드
            ret_val_ids = []
            if stmt.value:
                ret_val_ids = self._parse_expr(stmt.value, graph)
                created_node_ids.extend(ret_val_ids)

            term_node_id = self._generate_node_id("N_PY_RET")
            inv_sig = "OP_TERMINATE_EMIT:Name" if isinstance(stmt.value, ast.Name) else INVARIANT_TERMINATE_EMIT

            node = CausalNode(
                node_id=term_node_id,
                node_type=NodeType.STEM,
                invariant_signature=inv_sig,
                payload={"return_type": type(stmt.value).__name__}
            )
            graph.add_node(node)
            graph.context_constraints[term_node_id].add(f"py_lineno:{stmt.lineno}")
            graph.context_constraints[term_node_id].add("py_syntax:ast.Return")

            for v_id in ret_val_ids:
                graph.add_edge(CausalEdge(
                    source_id=v_id,
                    target_id=term_node_id,
                    precondition="seq_flow",
                    is_necessary=True
                ))

            if incoming_seq_id and not ret_val_ids:
                graph.add_edge(CausalEdge(
                    source_id=incoming_seq_id,
                    target_id=term_node_id,
                    precondition="seq_flow",
                    is_necessary=True
                ))

            created_node_ids.append(term_node_id)

        elif isinstance(stmt, ast.Expr):
            expr_ids = self._parse_expr(stmt.value, graph)
            created_node_ids.extend(expr_ids)

        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            curr_seq = incoming_seq_id
            for sub_stmt in stmt.body:
                sub_ids = self._parse_statement(sub_stmt, graph, curr_seq)
                if sub_ids:
                    curr_seq = sub_ids[-1]
                    created_node_ids.extend(sub_ids)

        elif isinstance(stmt, ast.If):
            test_ids = self._parse_expr(stmt.test, graph)
            created_node_ids.extend(test_ids)
            curr_seq = test_ids[-1] if test_ids else incoming_seq_id
            for sub_stmt in stmt.body:
                sub_ids = self._parse_statement(sub_stmt, graph, curr_seq)
                if sub_ids:
                    curr_seq = sub_ids[-1]
                    created_node_ids.extend(sub_ids)
            for sub_stmt in stmt.orelse:
                sub_ids = self._parse_statement(sub_stmt, graph, curr_seq)
                if sub_ids:
                    curr_seq = sub_ids[-1]
                    created_node_ids.extend(sub_ids)

        return created_node_ids

    def _parse_expr(self, expr: ast.expr, graph: CausalGraph) -> List[str]:
        node_id = self._generate_node_id("N_PY_EXPR")

        if isinstance(expr, ast.BinOp):
            # 이항 연산
            op_sig = INVARIANT_OP_ADD
            if isinstance(expr.op, ast.Mult):
                op_sig = INVARIANT_OP_MULTIPLY
            elif isinstance(expr.op, ast.Sub):
                op_sig = INVARIANT_OP_SUBTRACT
            elif isinstance(expr.op, ast.Div):
                op_sig = INVARIANT_OP_DIVIDE

            left_ids = self._parse_expr(expr.left, graph)
            right_ids = self._parse_expr(expr.right, graph)

            node = CausalNode(
                node_id=node_id,
                node_type=NodeType.STEM,
                invariant_signature=op_sig,
                payload={"op_type": type(expr.op).__name__}
            )
            graph.add_node(node)
            graph.context_constraints[node_id].add(f"py_syntax:ast.BinOp:{type(expr.op).__name__}")

            for lid in left_ids:
                graph.add_edge(CausalEdge(source_id=lid, target_id=node_id, precondition="left_operand"))
            for rid in right_ids:
                graph.add_edge(CausalEdge(source_id=rid, target_id=node_id, precondition="right_operand"))

            return [node_id]

        elif isinstance(expr, ast.Constant):
            node = CausalNode(
                node_id=node_id,
                node_type=NodeType.STEM,
                invariant_signature=INVARIANT_CONST_VALUE,
                payload={"value": expr.value}
            )
            graph.add_node(node)
            graph.context_constraints[node_id].add(f"py_type:{type(expr.value).__name__}")
            return [node_id]

        elif isinstance(expr, ast.Name):
            node = CausalNode(
                node_id=node_id,
                node_type=NodeType.STEM,
                invariant_signature=INVARIANT_VAR_BINDING,
                payload={"var_id": expr.id}
            )
            graph.add_node(node)
            graph.context_constraints[node_id].add(f"py_identifier:{expr.id}")
            return [node_id]

        else:
            node = CausalNode(
                node_id=node_id,
                node_type=NodeType.STEM,
                invariant_signature=f"INVARIANT_OP_{type(expr).__name__.upper()}",
                payload={"raw_ast": type(expr).__name__}
            )
            graph.add_node(node)
            return [node_id]
