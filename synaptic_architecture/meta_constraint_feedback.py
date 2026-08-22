"""
[Meta-Constraint Feedback Loop & Re-perception Engine]
This module bridges lower-level fast-clock candidate generation with upper-level slow-clock meta-constraint control.
It re-inputs generated candidate trajectories as sensory perception into the Preisach Causal Field,
measures structural impedance (trajectory curvature, topological phase discrepancy, latency friction),
and performs Rule Mutation on the state-space constraints.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import causal_engine as ce
from core.physics.causal_engine import CausalEngine, CausalNode, TransitionRule, CausalState, StateDelta, AtomicAction


class MetaConstraintFeedbackLoop:
    """
    [MetaConstraintFeedbackLoop - 인과적 환류 및 재인식 엔지]
    1. 무제한 가상 조합(Fast Clock Exploration) 생성
    2. 생성물 -> C++ Preisach SoA 필드 u(t) 재입력 (Re-perception)
    3. 궤적 꺾임(Curvature) & 위상 불일치(Phase Discrepancy) -> 임피던스(마찰) 측정
    4. 상위 관조 로터(Slow Latency Damping) -> 제약 규칙 변형 (Rule Mutation)
    """

    def __init__(
        self,
        num_field_nodes: int = 64,
        hysterons_per_dim: int = 8,
        gamma_curvature: float = 0.3,
        latency_damping: float = 0.2,
        friction_threshold: float = 0.45,
    ):
        self.field = ce.PreisachTensorFieldSoA(num_field_nodes, hysterons_per_dim)
        self.extractor = ce.AttractorExtractionLayer()
        self.backtracer = ce.CausalBacktracer()
        self.closed_loop = ce.ClosedLoopCausalEngine()
        self.mutator = ce.MetaConstraintMutator()

        self.gamma_curvature = gamma_curvature
        self.latency_damping = latency_damping
        self.friction_threshold = friction_threshold

        self.history_impedance: List[ce.ImpedanceResult] = []

    def generate_candidate_trajectories(
        self,
        start_node_id: int,
        goal_node_id: int,
        num_candidates: int = 5,
        density_threshold: float = 0.35,
    ) -> Tuple[List[ce.MacroSymbolNode], List[ce.CausalEdge], List[List[int]]]:
        """
        [1단계: 무제한 후보 궤적 생성]
        현재 메타 제약 조건(MetaConstraintRule) 하에서 상태 공간을 탐색하여 가상 조합 궤적들을 구합니다.
        """
        # C++ Preisach Field Update
        ce.update_preisach_field(self.field)

        # Attractor Extraction
        raw_nodes, raw_edges = self.extractor.extract_causal_graph(self.field, density_threshold)

        # Apply Current Meta-Constraint Filtering (Rule A or A')
        filtered_nodes = self.mutator.filter_nodes(raw_nodes)
        filtered_edges = self.mutator.filter_edges(raw_edges)

        if not filtered_nodes or start_node_id >= len(filtered_nodes) or goal_node_id >= len(filtered_nodes):
            # Fallback path if nodes are scarce
            return filtered_nodes, filtered_edges, [[start_node_id, goal_node_id]]

        candidate_trajectories = []
        # Main path via A* backtracer
        main_path = self.backtracer.trace_minimal_impedance_path_with_latency(
            goal_node_id, start_node_id, filtered_nodes, filtered_edges, self.gamma_curvature, self.latency_damping
        )
        candidate_trajectories.append(main_path)

        # Generate variations (stochastic/perturbed exploration trajectories)
        for c in range(1, num_candidates):
            perturbed_path = list(main_path)
            if len(perturbed_path) > 2:
                # Inject structural fluctuation in inner nodes
                mid_idx = len(perturbed_path) // 2
                alt_node = (perturbed_path[mid_idx] + c) % len(filtered_nodes)
                perturbed_path[mid_idx] = alt_node
            candidate_trajectories.append(perturbed_path)

        return filtered_nodes, filtered_edges, candidate_trajectories

    def reperception_and_evaluate(
        self,
        nodes: List[ce.MacroSymbolNode],
        candidate_trajectory: List[int],
        target_trajectory: List[int],
    ) -> ce.ImpedanceResult:
        """
        [2단계 & 3단계: 재인식(Re-perception) 및 임피던스(마찰) 측정]
        생성된 궤적을 다시 Preisach SoA u(t) 입력으로 재투입한 뒤,
        위상 불일치와 궤적 꺾임(Curvature) 및 느린 댐퍼가 가미된 구조적 마찰을 산출합니다.
        """
        if candidate_trajectory and nodes:
            # Prepare signal vector from trajectory nodes
            input_signal = np.zeros(self.field.num_nodes, dtype=np.float32)
            for idx in candidate_trajectory:
                if idx < len(nodes):
                    node_field_target = idx % self.field.num_nodes
                    input_signal[node_field_target] += nodes[idx].pivot_alpha

            # Fast injection & Causal Field Re-perception Update
            self.field.set_input_signals_from_numpy(input_signal)
            ce.update_preisach_field(self.field)

        # Evaluate Structural Impedance
        impedance_result = ce.CausalImpedanceEvaluator.evaluate_impedance(
            nodes,
            candidate_trajectory,
            target_trajectory,
            self.gamma_curvature,
            self.latency_damping,
            self.friction_threshold,
        )

        self.history_impedance.append(impedance_result)
        return impedance_result

    def step_meta_feedback(
        self,
        start_node_id: int,
        goal_node_id: int,
        target_trajectory: List[int],
    ) -> Dict[str, Any]:
        """
        [4단계: 메타 피드백 루프 (Rule Mutation Execute)]
        1회 환류 루프: 후보 생성 -> 재인식/임피던스 평가 -> 상위 댐퍼 -> 제약조건 자가 변형.
        """
        nodes, edges, candidates = self.generate_candidate_trajectories(start_node_id, goal_node_id)
        best_candidate = candidates[0]

        impedance = self.reperception_and_evaluate(nodes, best_candidate, target_trajectory)

        rule_mutated = False
        if impedance.requires_rule_mutation:
            # Meta-Constraint Rule Mutation (Constraint A -> Constraint A')
            self.mutator.mutate_rule(impedance, nodes, best_candidate)
            rule_mutated = True

        current_rule = self.mutator.get_current_rule()

        return {
            "nodes_count": len(nodes),
            "edges_count": len(edges),
            "best_trajectory": best_candidate,
            "target_trajectory": target_trajectory,
            "trajectory_curvature": impedance.trajectory_curvature,
            "topological_phase_diff": impedance.topological_phase_diff,
            "latency_damped_friction": impedance.latency_damped_friction,
            "resonance_score": impedance.resonance_score,
            "rule_mutated": rule_mutated,
            "mutation_count": self.mutator.get_mutation_count(),
            "rule": {
                "max_reluctance": current_rule.max_reluctance_threshold,
                "min_rigidity": current_rule.min_rigidity_threshold,
                "alpha_bounds": (current_rule.alpha_boundary_min, current_rule.alpha_boundary_max),
                "beta_bounds": (current_rule.beta_boundary_min, current_rule.beta_boundary_max),
                "curvature_penalty": current_rule.curvature_penalty_weight,
            },
        }
