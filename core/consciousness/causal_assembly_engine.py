"""
Elysia Causal Assembly Framework
================================
Implements the 3 Causal Assembly Pillars:
1. Fixed Edge Dependency Mapping (고정된 엣지 매기기):
   External causal dependency map (A -> B) over existing fragmented resources, code, APIs, and data.
2. Static State Matrix Lookup (정적 룩업 및 매트릭스로 치환):
   O(1) pre-validated static state matrix replacing dynamic runtime condition branching.
3. Retroactive Intent Anchor & Causal Feedback Loop (의도 축 & 피드백 루프):
   Retroactively derives intent from execution outcomes, measures phase divergence (ΔΦ),
   and applies targeted variable feedback to align the system within the causal gravity field.
"""

import math
import numpy as np
from typing import Dict, Any, List, Callable, Optional, Tuple
from core.utils.math_utils import Quaternion, traverse_causal_trajectory


class FixedEdgeDependencyMap:
    """
    [Pillar 1: Fixed Edge Dependency Mapping]
    Wraps existing fragmented resources (functions, data sources, APIs)
    and binds them with explicit causal dependency edges (A -> B).

    "What changes in A must deterministically transform B."
    """
    def __init__(self):
        self.nodes: Dict[str, Callable[..., Any]] = {}
        self.dependencies: Dict[str, List[str]] = {} # node_id -> list of dependent_node_ids
        self.reverse_deps: Dict[str, List[str]] = {}  # node_id -> list of prerequisite_node_ids
        self.last_execution_state: Dict[str, Any] = {}

    def register_fragment(self, node_id: str, func: Callable[..., Any]):
        """Registers a code, API, or data fragment without altering its internals."""
        self.nodes[node_id] = func
        if node_id not in self.dependencies:
            self.dependencies[node_id] = []
        if node_id not in self.reverse_deps:
            self.reverse_deps[node_id] = []

    def add_causal_edge(self, source_id: str, target_id: str):
        """
        Fixes a directed causal edge: source_id -> target_id.
        When source_id updates/changes, target_id MUST be re-evaluated.
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            raise KeyError(f"Both nodes must be registered: {source_id}, {target_id}")

        if target_id not in self.dependencies[source_id]:
            self.dependencies[source_id].append(target_id)
        if source_id not in self.reverse_deps[target_id]:
            self.reverse_deps[target_id].append(source_id)

    def get_execution_order(self, start_nodes: Optional[List[str]] = None) -> List[str]:
        """
        Topological sort to determine deterministic execution order of fragments.
        """
        nodes_to_process = start_nodes if start_nodes else list(self.nodes.keys())
        visited = set()
        order = []

        def visit(n: str):
            if n not in visited:
                visited.add(n)
                # Process prerequisites first
                for prereq in self.reverse_deps.get(n, []):
                    if prereq in nodes_to_process:
                        visit(prereq)
                order.append(n)

        for node in nodes_to_process:
            visit(node)

        return order

    def propagate(self, initial_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propagates data through the fixed causal dependency graph.
        """
        order = self.get_execution_order()
        results = dict(initial_inputs)

        for node_id in order:
            if node_id in self.nodes:
                func = self.nodes[node_id]
                # Gather inputs from prerequisites or initial_inputs
                prereqs = self.reverse_deps.get(node_id, [])
                if prereqs:
                    kwargs = {p: results.get(p) for p in prereqs}
                    res = func(**kwargs)
                else:
                    if node_id in results:
                        res = func(results[node_id])
                    else:
                        res = func()
                results[node_id] = res

        self.last_execution_state = results
        return results


class StaticStateMatrix:
    """
    [Pillar 2: Static State Matrix & Lookup]
    Replaces runtime dynamic branching (if-else condition evaluations)
    with a pre-validated, frozen state matrix and lookup table.

    Variables pass through pre-calculated pathways without runtime calculation overhead.
    """
    def __init__(self, key_dim: int = 4):
        self.key_dim = key_dim
        self.matrix_rules: List[Dict[str, Any]] = []
        self.lookup_table: Dict[str, Any] = {}

    def register_pathway(self, key_pattern: str, state_vector: np.ndarray, verified_outcome: Any):
        """
        Registers a verified input-output pathway in the static state matrix.
        """
        self.lookup_table[key_pattern] = {
            "state_vector": state_vector,
            "outcome": verified_outcome
        }
        self.matrix_rules.append({
            "key": key_pattern,
            "vector": state_vector,
            "outcome": verified_outcome
        })

    def quantize_input(self, variables: np.ndarray) -> str:
        """
        Quantizes input variables into a discrete lookup key for O(1) pathway retrieval.
        """
        quantized = np.sign(variables).astype(int)
        return "_".join(map(str, quantized))

    def evaluate(self, variable_inputs: np.ndarray) -> Tuple[Any, float]:
        """
        Passes variables through the static state matrix.
        Returns (outcome, friction_cost).
        Zero runtime condition branching: pure matrix projection / lookup.
        """
        key = self.quantize_input(variable_inputs)

        # O(1) direct lookup if exact pattern matched
        if key in self.lookup_table:
            entry = self.lookup_table[key]
            return entry["outcome"], 0.0

        # Vectorized matrix distance projection fallback
        if not self.matrix_rules:
            return None, 1.0

        vectors = np.array([r["vector"] for r in self.matrix_rules])
        norm_input = variable_inputs / (np.linalg.norm(variable_inputs) + 1e-9)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        norm_vectors = vectors / norms

        resonances = np.dot(norm_vectors, norm_input)
        best_idx = int(np.argmax(resonances))
        best_rule = self.matrix_rules[best_idx]

        friction = 1.0 - float(max(0.0, resonances[best_idx]))
        return best_rule["outcome"], friction


class RetroactiveIntentAnchor:
    """
    [Pillar 3: Retroactive Intent Anchor & Feedback Loop]
    Maintains a top-level Intent Anchor derived retroactively from actual execution outcomes.

    "Intent is not a mystical priori blueprint, but a retroactive derivation
    from the actual executed outcome within the causal field."
    """
    def __init__(self, initial_anchor_vector: Optional[Quaternion] = None):
        self.intent_anchor: Quaternion = initial_anchor_vector or Quaternion(1.0, 0.0, 0.0, 0.0)
        self.history: List[Dict[str, Any]] = []

    def derive_intent_from_outcome(self, outcome_data: Any) -> Quaternion:
        """
        Retroactively extracts the intent trajectory (Quaternion) from execution results.
        """
        if isinstance(outcome_data, bytes):
            raw_bytes = outcome_data
        elif isinstance(outcome_data, str):
            raw_bytes = outcome_data.encode("utf-8")
        elif isinstance(outcome_data, (dict, list)):
            raw_bytes = str(outcome_data).encode("utf-8")
        elif isinstance(outcome_data, np.ndarray):
            raw_bytes = outcome_data.tobytes()
        else:
            raw_bytes = str(outcome_data).encode("utf-8")

        return traverse_causal_trajectory(raw_bytes)

    def measure_phase_divergence(self, actual_outcome_q: Quaternion) -> float:
        """
        Measures the phase divergence (ΔΦ) / tension between the actual outcome and the Intent Anchor.
        ΔΦ = 1.0 - |q_intent · q_outcome|
        """
        dot = abs(self.intent_anchor.dot(actual_outcome_q))
        return float(1.0 - min(1.0, dot))

    def update_intent_anchor(self, actual_outcome_q: Quaternion, adaptation_rate: float = 0.2):
        """
        Retroactively shifts the Intent Anchor toward the newly realized outcome trajectory.
        """
        self.intent_anchor = Quaternion.slerp(self.intent_anchor, actual_outcome_q, adaptation_rate)

    def generate_variable_feedback(self, variable_inputs: np.ndarray, actual_outcome_q: Quaternion) -> Tuple[np.ndarray, float]:
        """
        Calculates feedback adjustment for variables based on phase divergence (ΔΦ).
        Aligns variables to restore trajectory back to the Intent Anchor gravity field.
        """
        phase_divergence = self.measure_phase_divergence(actual_outcome_q)

        if phase_divergence < 1e-4:
            return variable_inputs, 0.0

        q_err = self.intent_anchor * actual_outcome_q.conjugate()
        axis = q_err.axis
        angle = q_err.angle

        adjustment_factor = min(1.0, phase_divergence * 2.0)

        correction = np.zeros_like(variable_inputs, dtype=float)
        dim = min(len(variable_inputs), len(axis))
        correction[:dim] = axis[:dim] * (angle * adjustment_factor)

        adjusted_variables = variable_inputs + correction

        self.history.append({
            "phase_divergence": phase_divergence,
            "adjustment_norm": float(np.linalg.norm(correction)),
            "aligned": phase_divergence < 0.1
        })

        return adjusted_variables, phase_divergence


class CausalAssemblyEngine:
    """
    [Unified Causal Assembly Engine]
    Combines Dependency Mapping, Static State Matrix, and Retroactive Intent Anchor
    to force heterogeneous fragments into a cohesive, deterministic causal mechanism.
    """
    def __init__(self, key_dim: int = 4):
        self.dependency_map = FixedEdgeDependencyMap()
        self.state_matrix = StaticStateMatrix(key_dim=key_dim)
        self.intent_anchor = RetroactiveIntentAnchor()

    def assemble_fragments(
        self,
        fragments: Dict[str, Callable[..., Any]],
        causal_edges: List[Tuple[str, str]],
        static_pathways: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Reassembles fragmented modules and APIs into a unified causal graph.
        """
        for node_id, func in fragments.items():
            self.dependency_map.register_fragment(node_id, func)

        for source_id, target_id in causal_edges:
            self.dependency_map.add_causal_edge(source_id, target_id)

        if static_pathways:
            for p in static_pathways:
                self.state_matrix.register_pathway(
                    key_pattern=p["key"],
                    state_vector=np.array(p["vector"], dtype=float),
                    verified_outcome=p["outcome"]
                )

    def run_causal_cycle(self, initial_data: Dict[str, Any], variable_inputs: np.ndarray) -> Dict[str, Any]:
        """
        Executes a full causal cycle:
        1. Propagate data across fixed dependency map edges.
        2. Evaluate static matrix lookup for deterministic fast path.
        3. Extract retroactive intent from actual outcome.
        4. Measure phase divergence and compute variable feedback adjustment.
        """
        propagated_results = self.dependency_map.propagate(initial_data)

        matrix_outcome, matrix_friction = self.state_matrix.evaluate(variable_inputs)

        combined_outcome = {
            "graph_results": propagated_results,
            "matrix_outcome": matrix_outcome,
            "matrix_friction": matrix_friction
        }

        outcome_q = self.intent_anchor.derive_intent_from_outcome(combined_outcome)

        adjusted_variables, phase_divergence = self.intent_anchor.generate_variable_feedback(
            variable_inputs, outcome_q
        )

        return {
            "propagated_results": propagated_results,
            "matrix_outcome": matrix_outcome,
            "matrix_friction": matrix_friction,
            "outcome_quaternion": outcome_q.elements,
            "phase_divergence": phase_divergence,
            "adjusted_variables": adjusted_variables,
            "intent_aligned": phase_divergence < 0.1
        }
