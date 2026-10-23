"""
ELYSIA Relational Structural Coupling & Proof Protocol Engine v0.2

This module implements the 4-Step Structural Coupling & Discrepancy Tracking Experiment:
1. Step 1 (Existing Relation Inference & Proof Protocol):
   Records relational graph (e.g. 1 -> 2 -> 3) and internal proof justification protocol.
2. Step 2 (Prediction & Discrepancy Classification):
   Generates future predictions based on current relational topology and evaluates incoming observations.
   Distinguishes between scalar numeric noise (small phase/value drift) vs. structural discrepancies (unexpected trajectory e.g. 1 -> 2 -> 4).
3. Step 3 (Structural Reactions):
   Executes 4 distinct structural reactions based on interaction conditions and proof constraints:
   - MAINTAIN: Ignore insignificant noise; keep relational structure intact.
   - MODIFY: Update edge coupling weights/rules when structural continuity holds.
   - BRANCH (Mitosis): Split into parallel contextual topological layers when encountering mutually exclusive rules or context shifts.
   - SUSPEND_JUDGMENT: Freeze belief mutation and preserve uncertainty when data is insufficient or ambiguous.
4. Step 4 (Re-observation & Proof Protocol Evolution):
   Demonstrates that structural changes actively alter future predictions and proof justifications in a closed loop.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import uuid


@dataclass
class RelationalNode:
    """Node representing an entity or state in the relational network."""
    node_id: str
    label: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationalEdge:
    """Directed relational link expressing structural causality/transition."""
    source_id: str
    target_id: str
    relation_type: str = "TRANSITION"  # TRANSITION, CAUSAL, EQUIVALENCE, CONSTRAINED
    weight: float = 1.0
    context_condition: Optional[str] = None


@dataclass
class ProofStep:
    """Step in the internal justification protocol documenting how an inference was derived."""
    step_index: int
    rule_applied: str
    premise_node_ids: List[str]
    conclusion_node_id: str
    justification_notes: str


@dataclass
class ProofProtocol:
    """Internal proof protocol observing and documenting the inference procedure."""
    protocol_id: str = field(default_factory=lambda: f"proof_{uuid.uuid4().hex[:8]}")
    steps: List[ProofStep] = field(default_factory=list)

    def add_step(self, rule_applied: str, premise_node_ids: List[str], conclusion_node_id: str, notes: str):
        step_idx = len(self.steps) + 1
        self.steps.append(ProofStep(step_idx, rule_applied, premise_node_ids, conclusion_node_id, notes))


class StructuralRelationalGraph:
    """
    Graph representing relational topology and structural connections between states/entities.
    Supports branching (Mitosis / Layer Separation) when mutually exclusive rules are encountered.
    """

    def __init__(self, graph_id: str = "main_layer"):
        self.graph_id = graph_id
        self.nodes: Dict[str, RelationalNode] = {}
        self.edges: Dict[Tuple[str, str], RelationalEdge] = {}
        self.proof_protocol = ProofProtocol()

    def add_node(self, node_id: str, label: str, attributes: Optional[Dict[str, Any]] = None):
        if node_id not in self.nodes:
            self.nodes[node_id] = RelationalNode(node_id, label, attributes or {})

    def add_edge(self, source_id: str, target_id: str, relation_type: str = "TRANSITION", weight: float = 1.0, context: Optional[str] = None):
        self.add_node(source_id, source_id)
        self.add_node(target_id, target_id)
        edge = RelationalEdge(source_id, target_id, relation_type, weight, context)
        self.edges[(source_id, target_id)] = edge

        self.proof_protocol.add_step(
            rule_applied="ADD_RELATIONAL_EDGE",
            premise_node_ids=[source_id],
            conclusion_node_id=target_id,
            notes=f"Linked {source_id} -> {target_id} under relation '{relation_type}' (weight={weight}, context={context})"
        )

    def get_successors(self, node_id: str) -> List[Tuple[str, RelationalEdge]]:
        return [(target, edge) for (src, target), edge in self.edges.items() if src == node_id]

    def predict_next_state(self, current_node_id: str) -> Optional[str]:
        """Predicts the next expected state based on current relational graph edges."""
        successors = self.get_successors(current_node_id)
        if not successors:
            return None
        # Pick successor with highest weight
        successors.sort(key=lambda x: x[1].weight, reverse=True)
        predicted_target = successors[0][0]

        self.proof_protocol.add_step(
            rule_applied="INFER_NEXT_STATE",
            premise_node_ids=[current_node_id],
            conclusion_node_id=predicted_target,
            notes=f"Predicted transition from {current_node_id} -> {predicted_target} via edge weight {successors[0][1].weight}"
        )
        return predicted_target

    def clone_branch(self, new_layer_id: str) -> 'StructuralRelationalGraph':
        """Executes topological Mitosis / Layer Separation for parallel context management."""
        branch_graph = StructuralRelationalGraph(graph_id=new_layer_id)
        for n_id, node in self.nodes.items():
            branch_graph.nodes[n_id] = RelationalNode(node.node_id, node.label, node.attributes.copy())
        for key, edge in self.edges.items():
            branch_graph.edges[key] = RelationalEdge(edge.source_id, edge.target_id, edge.relation_type, edge.weight, edge.context_condition)
        branch_graph.proof_protocol = ProofProtocol(protocol_id=f"proof_{new_layer_id}")
        return branch_graph


@dataclass
class DiscrepancyAnalysis:
    discrepancy_type: str  # "NOISE", "STRUCTURAL_DISCREPANCY", "MUTUALLY_EXCLUSIVE_CONFLICT", "AMBIGUOUS_INSUFFICIENT"
    magnitude: float
    observed_transition: Tuple[str, str]
    expected_transition: Optional[Tuple[str, str]]
    explanation: str


class StructuralCouplingEngine:
    """
    Engine executing the 4-Step Structural Coupling & Discrepancy Tracking Experiment:
    1. Relation Inference
    2. Discrepancy Classification (Noise vs Structural Discrepancy)
    3. Structural Reaction (Maintain, Modify, Branch, Suspend Judgment)
    4. Re-observation & Future Prediction Protocol Evolution
    """

    def __init__(self, noise_threshold: float = 0.15, ambiguity_threshold: float = 0.8):
        self.noise_threshold = noise_threshold
        self.ambiguity_threshold = ambiguity_threshold
        self.active_layers: Dict[str, StructuralRelationalGraph] = {"main_layer": StructuralRelationalGraph("main_layer")}
        self.primary_layer_id: str = "main_layer"
        self.reaction_history: List[Dict[str, Any]] = []

    @property
    def primary_graph(self) -> StructuralRelationalGraph:
        return self.active_layers[self.primary_layer_id]

    def record_initial_relation(self, sequence: List[str]):
        """Step 1: Records initial relational path sequence (e.g., 1 -> 2 -> 3)."""
        graph = self.primary_graph
        for i in range(len(sequence) - 1):
            graph.add_edge(sequence[i], sequence[i+1], relation_type="TRANSITION", weight=1.0)

    def analyze_observation_discrepancy(
        self,
        current_state: str,
        observed_next_state: str,
        observed_value_delta: float = 0.0,
        context_condition: Optional[str] = None,
        is_ambiguous_data: bool = False
    ) -> DiscrepancyAnalysis:
        """
        Step 2: Analyzes incoming observation vs. internal prediction.
        Distinguishes scalar numeric noise from structural relational discrepancies.
        """
        graph = self.primary_graph
        expected_next_state = graph.predict_next_state(current_state)

        if is_ambiguous_data:
            return DiscrepancyAnalysis(
                discrepancy_type="AMBIGUOUS_INSUFFICIENT",
                magnitude=observed_value_delta,
                observed_transition=(current_state, observed_next_state),
                expected_transition=(current_state, expected_next_state) if expected_next_state else None,
                explanation="Observational data is ambiguous or insufficient to establish structural relation."
            )

        if expected_next_state == observed_next_state:
            if abs(observed_value_delta) <= self.noise_threshold:
                return DiscrepancyAnalysis(
                    discrepancy_type="NOISE",
                    magnitude=observed_value_delta,
                    observed_transition=(current_state, observed_next_state),
                    expected_transition=(current_state, expected_next_state),
                    explanation=f"Observation matches expected structure ({current_state}->{observed_next_state}). Value delta {observed_value_delta} is within noise threshold."
                )
            else:
                return DiscrepancyAnalysis(
                    discrepancy_type="STRUCTURAL_DISCREPANCY",
                    magnitude=observed_value_delta,
                    observed_transition=(current_state, observed_next_state),
                    expected_transition=(current_state, expected_next_state),
                    explanation=f"State matches ({current_state}->{observed_next_state}), but large value perturbation ({observed_value_delta}) indicates structural property modification."
                )

        # Expected state differs from observed state (e.g. expected 3, observed 4)
        if context_condition and context_condition != "default":
            return DiscrepancyAnalysis(
                discrepancy_type="MUTUALLY_EXCLUSIVE_CONFLICT",
                magnitude=1.0,
                observed_transition=(current_state, observed_next_state),
                expected_transition=(current_state, expected_next_state) if expected_next_state else None,
                explanation=f"Observed transition ({current_state}->{observed_next_state}) under context '{context_condition}' conflicts with existing rule ({current_state}->{expected_next_state})."
            )

        return DiscrepancyAnalysis(
            discrepancy_type="STRUCTURAL_DISCREPANCY",
            magnitude=1.0,
            observed_transition=(current_state, observed_next_state),
            expected_transition=(current_state, expected_next_state) if expected_next_state else None,
            explanation=f"Observed transition ({current_state}->{observed_next_state}) differs structurally from expected transition ({current_state}->{expected_next_state})."
        )

    def process_observation(
        self,
        current_state: str,
        observed_next_state: str,
        observed_value_delta: float = 0.0,
        context_condition: Optional[str] = None,
        is_ambiguous_data: bool = False
    ) -> Dict[str, Any]:
        """
        Step 3 & 4: Evaluates discrepancy and executes one of the 4 Structural Reactions:
        - MAINTAIN: Keep structure untouched.
        - MODIFY: Adjust relational weights/edges.
        - BRANCH (Mitosis): Layer separation into parallel context graph.
        - SUSPEND_JUDGMENT: Hold belief mutation in uncertainty.
        """
        analysis = self.analyze_observation_discrepancy(
            current_state, observed_next_state, observed_value_delta, context_condition, is_ambiguous_data
        )

        reaction = "MAINTAIN"
        details = {}

        if analysis.discrepancy_type == "NOISE":
            reaction = "MAINTAIN"
            details["action"] = "Retained existing relational structure without mutation."
            self.primary_graph.proof_protocol.add_step(
                rule_applied="REACTION_MAINTAIN",
                premise_node_ids=[current_state],
                conclusion_node_id=observed_next_state,
                notes=f"MAINTAIN: Discrepancy classified as noise ({observed_value_delta:.3f}). Relational graph preserved."
            )

        elif analysis.discrepancy_type == "AMBIGUOUS_INSUFFICIENT":
            reaction = "SUSPEND_JUDGMENT"
            details["action"] = "Judgment suspended; belief mutation frozen due to ambiguity/insufficient evidence."
            self.primary_graph.proof_protocol.add_step(
                rule_applied="REACTION_SUSPEND_JUDGMENT",
                premise_node_ids=[current_state],
                conclusion_node_id=observed_next_state,
                notes="SUSPEND_JUDGMENT: Insufficient evidence to mutate relational topology. Decision deferred."
            )

        elif analysis.discrepancy_type == "MUTUALLY_EXCLUSIVE_CONFLICT":
            reaction = "BRANCH"
            new_layer_id = f"context_{context_condition}"
            branch_graph = self.primary_graph.clone_branch(new_layer_id)
            branch_graph.add_edge(current_state, observed_next_state, relation_type="TRANSITION", weight=1.5, context=context_condition)
            self.active_layers[new_layer_id] = branch_graph

            details["action"] = f"Topological Mitosis executed. Created parallel layer '{new_layer_id}' for context '{context_condition}'."
            details["new_layer_id"] = new_layer_id

            self.primary_graph.proof_protocol.add_step(
                rule_applied="REACTION_BRANCH_MITOSIS",
                premise_node_ids=[current_state],
                conclusion_node_id=observed_next_state,
                notes=f"BRANCH (Mitosis): Mutually exclusive rule detected in context '{context_condition}'. Created parallel branch graph '{new_layer_id}'."
            )

        elif analysis.discrepancy_type == "STRUCTURAL_DISCREPANCY":
            reaction = "MODIFY"
            # Add or update edge to accommodate new structural trajectory
            self.primary_graph.add_edge(current_state, observed_next_state, relation_type="TRANSITION", weight=1.2, context=context_condition)

            # Decay old edge if present
            if (current_state, analysis.expected_transition[1]) in self.primary_graph.edges if analysis.expected_transition else False:
                self.primary_graph.edges[(current_state, analysis.expected_transition[1])].weight *= 0.5

            details["action"] = f"Modified relational graph. Added/reinforced transition ({current_state}->{observed_next_state})."
            self.primary_graph.proof_protocol.add_step(
                rule_applied="REACTION_MODIFY",
                premise_node_ids=[current_state],
                conclusion_node_id=observed_next_state,
                notes=f"MODIFY: Structural discrepancy detected ({current_state}->{observed_next_state}). Graph topology updated."
            )

        result_record = {
            "current_state": current_state,
            "observed_next_state": observed_next_state,
            "analysis": analysis,
            "reaction": reaction,
            "details": details,
            "primary_layer_id": self.primary_layer_id,
            "active_layer_count": len(self.active_layers)
        }
        self.reaction_history.append(result_record)
        return result_record

    def predict_future_under_layer(self, current_state: str, layer_id: str = "main_layer") -> Optional[str]:
        """Step 4: Predicts future transition using the specified relational topology layer."""
        graph = self.active_layers.get(layer_id, self.primary_graph)
        return graph.predict_next_state(current_state)
