"""
Causal Receptor & Deconstructor (Phase 1)
=========================================
Deconstructs external high-level symbols, expressions, source code (AST), and signal streams
into micro atomic causal nodes (Single Static Assignment - SSA form / Atomic Causal DAG).
Strips static semantic labels, retaining pure causal dependencies, topological interaction density,
and chromatic signatures.
"""

import ast
import dataclasses
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


@dataclasses.dataclass
class AtomicCausalNode:
    """Represents an atomic, non-decomposable cause-and-effect relationship node."""
    id: str
    op: str  # e.g., 'FETCH', 'ADD', 'MUL', 'SUB', 'DIV', 'ASSIGN', 'RELAY', 'TRIGGER'
    parents: List[str]  # IDs of immediate causal ancestor nodes
    children: List[str]  # IDs of immediate causal dependent nodes
    chromatic_signature: np.ndarray  # [Flux (Red), Order (Blue), Entropy (Yellow)]
    phase_state: float = 0.0  # Internal phase state [0, 2*pi)
    energy: float = 1.0
    value: Optional[Any] = None  # Scalar value or array if evaluated


class AtomicCausalGraph:
    """DAG of atomic causal nodes stripped of static symbol labels."""
    def __init__(self):
        self.nodes: Dict[str, AtomicCausalNode] = {}
        self.node_counter: int = 0
        self.entry_nodes: List[str] = []
        self.exit_nodes: List[str] = []

    def add_node(
        self,
        op: str,
        parents: Optional[List[str]] = None,
        chromatic_signature: Optional[np.ndarray] = None,
        value: Optional[Any] = None
    ) -> AtomicCausalNode:
        parents = parents or []
        node_id = f"atomic_node_{self.node_counter}"
        self.node_counter += 1

        if chromatic_signature is None:
            # Default chromatic signature: [Flux, Order, Entropy]
            if op in ("FETCH", "INPUT"):
                chromatic_signature = np.array([0.8, 0.1, 0.1], dtype=np.float32)  # High Flux
            elif op in ("ADD", "MUL", "MULT", "SUB", "DIV", "OP"):
                chromatic_signature = np.array([0.2, 0.7, 0.1], dtype=np.float32)  # High Order
            else:
                chromatic_signature = np.array([0.3, 0.3, 0.4], dtype=np.float32)

        node = AtomicCausalNode(
            id=node_id,
            op=op,
            parents=list(parents),
            children=[],
            chromatic_signature=chromatic_signature,
            value=value
        )
        self.nodes[node_id] = node

        # Link parents to this new child node
        for p_id in parents:
            if p_id in self.nodes:
                self.nodes[p_id].children.append(node_id)

        if not parents:
            self.entry_nodes.append(node_id)

        return node

    def get_adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Returns adjacency matrix representing causal connection density and node order."""
        node_ids = list(self.nodes.keys())
        n = len(node_ids)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        adj = np.zeros((n, n), dtype=np.float32)

        for src_id, node in self.nodes.items():
            src_idx = id_to_idx[src_id]
            for child_id in node.children:
                if child_id in id_to_idx:
                    dst_idx = id_to_idx[child_id]
                    adj[src_idx, dst_idx] = 1.0

        return adj, node_ids

    def compute_causal_distance_matrix(self) -> np.ndarray:
        """Computes shortest topological causal distance d(i, j) between all node pairs."""
        n = len(self.nodes)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)

        adj, node_ids = self.get_adjacency_matrix()

        # Floyd-Warshall for shortest causal path / distance
        dist = np.full((n, n), fill_value=np.inf, dtype=np.float32)
        np.fill_diagonal(dist, 0.0)

        for i in range(n):
            for j in range(n):
                if adj[i, j] > 0:
                    dist[i, j] = 1.0  # Causal step distance = 1.0

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]

        # Symmetric topological distance approximation for spatial layout mapping
        sym_dist = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                d_ij = dist[i, j]
                d_ji = dist[j, i]
                min_d = min(d_ij, d_ji)
                if np.isinf(min_d):
                    sym_dist[i, j] = 10.0  # Unconnected max topological distance
                else:
                    sym_dist[i, j] = min_d

        return sym_dist


class CausalReceptor:
    """Frontend Deconstructor converting high-level Python code/AST or signals into SSA Causal Graphs."""

    def deconstruct_code(self, source_code: str) -> AtomicCausalGraph:
        """Parses Python source code, converts into Single Static Assignment (SSA) form,

        and strips variable names to yield an AtomicCausalGraph.
        """
        parsed_ast = ast.parse(source_code)
        graph = AtomicCausalGraph()
        env: Dict[str, str] = {}  # Symbol table mapping variable name -> latest atomic node_id

        class SSAVisitor(ast.NodeVisitor):
            def visit_Assign(self, node: ast.Assign):
                # Process RHS expression
                rhs_node_id = self.visit_expr(node.value)
                # Assign to LHS targets
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = rhs_node_id

            def visit_expr(self, expr_node: ast.AST) -> str:
                if isinstance(expr_node, ast.BinOp):
                    left_id = self.visit_expr(expr_node.left)
                    right_id = self.visit_expr(expr_node.right)
                    op_type = type(expr_node.op).__name__.upper()
                    atomic_node = graph.add_node(op=op_type, parents=[left_id, right_id])
                    return atomic_node.id
                elif isinstance(expr_node, ast.UnaryOp):
                    operand_id = self.visit_expr(expr_node.operand)
                    op_type = type(expr_node.op).__name__.upper()
                    atomic_node = graph.add_node(op=op_type, parents=[operand_id])
                    return atomic_node.id
                elif isinstance(expr_node, ast.Name):
                    if expr_node.id in env:
                        return env[expr_node.id]
                    else:
                        # External variable input / fetch
                        atomic_node = graph.add_node(op="FETCH", parents=[])
                        env[expr_node.id] = atomic_node.id
                        return atomic_node.id
                elif isinstance(expr_node, ast.Constant):
                    atomic_node = graph.add_node(op="CONST", parents=[], value=expr_node.value)
                    return atomic_node.id
                elif isinstance(expr_node, ast.Call):
                    arg_ids = [self.visit_expr(arg) for arg in expr_node.args]
                    func_name = expr_node.func.id if isinstance(expr_node.func, ast.Name) else "CALL"
                    atomic_node = graph.add_node(op=f"CALL_{func_name.upper()}", parents=arg_ids)
                    return atomic_node.id
                else:
                    # Generic node fallback
                    atomic_node = graph.add_node(op="GENERIC_OP", parents=[])
                    return atomic_node.id

        visitor = SSAVisitor()
        for stmt in parsed_ast.body:
            visitor.visit(stmt)

        return graph

    def deconstruct_signal_stream(self, signals: np.ndarray, threshold: float = 0.1) -> AtomicCausalGraph:
        """Deconstructs continuous signal time-series into atomic causal event nodes."""
        graph = AtomicCausalGraph()
        prev_node_id: Optional[str] = None

        diffs = np.diff(signals, axis=0) if signals.ndim > 1 else np.diff(signals)

        for idx, delta in enumerate(diffs):
            magnitude = float(np.linalg.norm(delta))
            if magnitude > threshold:
                parents = [prev_node_id] if prev_node_id else []
                node = graph.add_node(
                    op="SIGNAL_EDGE",
                    parents=parents,
                    value=magnitude,
                    chromatic_signature=np.array([0.7, 0.2, 0.1], dtype=np.float32)
                )
                prev_node_id = node.id

        return graph
