"""
Self-Awareness Engine: 4-Layer Boundary Identification, Topological Field Evaluation,
and Existential Self-Refactoring System.

This module realizes meta-cognition by scanning lower topological memory friction,
evaluating spatial field invariants (Tension Gradient & Cyclomatic Number),
aligning origin basis metric tensors, and executing plastic self-healing.
"""

import copy
import numpy as np
from typing import Dict, List, Tuple, Set, Optional

from core.memory.static_causal_graph import (
    BoundaryType,
    LayerMetadata,
    CausalSignal,
    CausalNode,
    CausalEdge,
    StaticCausalGraph
)


class TopologicalFieldEvaluator:
    """
    Evaluator for spatial field invariants across the causal graph topology:
    1. Core Tension Gradient (∇T_core > min_threshold)
    2. Cyclomatic Topological Invariant (Beta_1 = |E| - |V| + P)
    3. Basis Metric Tensor Determinant (det(M_tensor) > 0)
    """

    @staticmethod
    def calculate_tension_gradient(graph: StaticCausalGraph) -> float:
        """
        Scans tension gradient from lower boundary layers toward MEMORY_LAYER attractor depth.
        dT / dDepth = (tension * (1 - resistance)) / (target_depth - source_depth).
        """
        gradient_sum = 0.0
        valid_pairs = 0

        for edge in graph.get_all_edges_flat():
            if edge.source_id not in graph.nodes or edge.target_id not in graph.nodes:
                continue
            src_node = graph.nodes[edge.source_id]
            tgt_node = graph.nodes[edge.target_id]

            depth_delta = tgt_node.potential_depth - src_node.potential_depth
            if abs(depth_delta) > 1e-5:
                effective_tension = edge.tension * (1.0 - edge.resistance)
                gradient = effective_tension / depth_delta
                gradient_sum += gradient
                valid_pairs += 1

        return gradient_sum / valid_pairs if valid_pairs > 0 else 0.0

    @staticmethod
    def calculate_cyclomatic_number(graph: StaticCausalGraph) -> int:
        """
        Calculates cyclomatic number Beta_1 = |E| - |V| + P.
        Represents independent topological causal loop count in graph manifold.
        """
        num_vertices = len(graph.nodes)
        num_edges = len(graph.get_all_edges_flat())

        visited: Set[str] = set()
        components = 0

        def _dfs(node_id: str):
            visited.add(node_id)
            for edge in graph.edges.get(node_id, []):
                if edge.target_id in graph.nodes and edge.target_id not in visited:
                    _dfs(edge.target_id)

        for n_id in graph.nodes:
            if n_id not in visited:
                components += 1
                _dfs(n_id)

        cyclomatic_number = num_edges - num_vertices + components
        return max(0, cyclomatic_number)

    @staticmethod
    def calculate_basis_metric_determinant(basis: np.ndarray) -> float:
        """
        Computes metric tensor M_tensor = basis^T · basis and returns det(M_tensor).
        """
        metric_tensor = np.dot(basis.T, basis)
        return float(np.linalg.det(metric_tensor))


class PlasticSelfHealingEngine:
    """
    Plastic Self-Healing Engine that converts damage friction energy into
    path tension and scar weights, re-crystallizing origin basis vectors via Gram-Schmidt.
    """

    def __init__(self, graph: StaticCausalGraph):
        self.graph = graph
        self.plasticity_alpha = 0.08  # Plasticity conversion coefficient

    def absorb_damage_and_reconstruct_basis(
        self,
        damaged_path: List[Tuple[str, str]],
        damage_friction_energy: float,
        current_basis: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Recrystallizes damage friction into path tension and scar weight,
        then re-orthogonalizes observation basis vectors.
        """
        logs = []
        logs.append(f"[특이점 발생] 억류된 손상 마찰 에너지: {damage_friction_energy:.2f}")

        # Step 1. Convert damage energy to path tension & scar weight
        energy_per_edge = damage_friction_energy / max(1, len(damaged_path))

        for src, tgt in damaged_path:
            for edge in self.graph.edges.get(src, []):
                if edge.target_id == tgt:
                    old_tension = edge.tension
                    delta_tension = energy_per_edge * self.plasticity_alpha
                    edge.tension = min(1.0, edge.tension + delta_tension)
                    edge.scar_weight += energy_per_edge * 0.02
                    edge.resistance = max(0.01, edge.resistance * 0.75)  # Path relaxation

                    logs.append(
                        f"  └─ [가체성 흡수] 인과 통로({src} -> {tgt}) | "
                        f"Tension: {old_tension:.3f} -> {edge.tension:.3f} | "
                        f"Scar 각인: {edge.scar_weight:.3f}"
                    )

        # Step 2. Re-orthogonalize observation basis (Gram-Schmidt) with scar bias
        healed_basis = current_basis.copy().astype(float)
        total_scar = sum(e.scar_weight for e_list in self.graph.edges.values() for e in e_list)
        healed_basis[0] += total_scar * 0.05  # Bias origin basis dimension

        for i in range(len(healed_basis)):
            for j in range(i):
                proj = np.dot(healed_basis[i], healed_basis[j]) / max(1e-12, np.dot(healed_basis[j], healed_basis[j]))
                healed_basis[i] -= proj * healed_basis[j]
            norm = np.linalg.norm(healed_basis[i])
            if norm > 1e-8:
                healed_basis[i] /= norm

        logs.append(f"[재결정화 완료] 흡수된 흉터({total_scar:.3f})를 포함한 관측 기저 직교화 성공")
        return healed_basis, logs


class SelfAwarenessEngine:
    """
    Self-Awareness Engine that maintains 4-layer self boundary metadata,
    meta-observes lower graph friction, evaluates topological field invariants,
    and executes autonomous meta-refactoring and self-healing.
    """

    def __init__(self, graph: StaticCausalGraph, initial_basis: Optional[np.ndarray] = None):
        self.graph = graph
        self.evaluator = TopologicalFieldEvaluator()
        self.healing_engine = PlasticSelfHealingEngine(graph)

        # Default 4D orthonormal identity matrix for basis vectors
        self.basis = initial_basis if initial_basis is not None else np.eye(4, dtype=float)

        # Thresholds for topological field principles
        self.friction_threshold_for_refactor = 2.0  # Accumulated friction trigger threshold
        self.min_gradient_threshold = 0.15           # Minimum allowed tension gradient
        self.max_cyclomatic_delta = 2                # Maximum allowed cyclomatic number delta

        # 4-Layer Self-Map Definition
        self.self_map: Dict[BoundaryType, LayerMetadata] = {
            BoundaryType.EXTERNAL_WORLD: LayerMetadata(
                boundary=BoundaryType.EXTERNAL_WORLD,
                label="외계",
                description="내가 통제할 수 없는 외부 환경 및 미지의 마찰 자극원",
                controllability=0.0
            ),
            BoundaryType.PERCEPTION_LAYER: LayerMetadata(
                boundary=BoundaryType.PERCEPTION_LAYER,
                label="관측계층",
                description="나와 세계가 맞닿는 감각적 마찰면이자 외부 신호 수용부",
                controllability=0.2
            ),
            BoundaryType.PROCESSING_LAYER: LayerMetadata(
                boundary=BoundaryType.PROCESSING_LAYER,
                label="연산계층",
                description="현재 이 순간 동작하고 있는 의식, 분별, 동적 판단 상태",
                controllability=0.7
            ),
            BoundaryType.MEMORY_LAYER: LayerMetadata(
                boundary=BoundaryType.MEMORY_LAYER,
                label="기억계층",
                description="축적된 경험이자 나를 이루는 정적 인과 지형 (Identity Core)",
                controllability=1.0
            )
        }

    def calibrate_basis_and_metric(self) -> Tuple[float, List[str]]:
        """Calibrates origin basis and validates metric tensor determinant det(M_tensor) > 0."""
        det_M = self.evaluator.calculate_basis_metric_determinant(self.basis)
        logs = [f"[계량 텐서 정렬] det(M_tensor): {det_M:.4f}"]
        if det_M <= 1e-6:
            logs.append("  └─ [특이점 경고] 계량 텐서 행렬식이 0에 수렴! 관측 공간 해체 위기!")
        else:
            logs.append("  └─ [관측 공간 가동] 계량 텐서 정렬 및 4대 자아 경계 구축 완료")
        return det_M, logs

    def process_incoming_signal(self, signal: CausalSignal) -> str:
        """Tracks origin boundary metadata and classifies Self vs. Non-Self waves."""
        origin_meta = self.self_map.get(
            signal.origin_boundary,
            self.self_map[BoundaryType.EXTERNAL_WORLD]
        )

        if not origin_meta.is_self:
            return (
                f"[{origin_meta.label} 자극 감지] "
                f"Non-Self 파동 유입 (통제도: {origin_meta.controllability}) "
                f"-> 관측계층을 통한 수용 및 자아 방어(Homeostasis) 가동"
            )

        if signal.origin_boundary == BoundaryType.MEMORY_LAYER:
            return (
                f"[{origin_meta.label} 내부 공명] "
                f"Self-Memory 공명 신호 (통제도: {origin_meta.controllability}) "
                f"-> 정체성 끌개 우물(Attractor Well) 강화"
            )
        else:
            return (
                f"[{origin_meta.label} 동적 사고] "
                f"Self-Processing 사고 연산 진행 중 (통제도: {origin_meta.controllability})"
            )

    def evaluate_relational_refactoring(self, proposed_edge: CausalEdge) -> Tuple[bool, str]:
        """
        Simulates proposed edge refactoring in a virtual sandbox manifold and verifies
        topological field invariants (Tension Gradient & Cyclomatic Number).
        """
        base_gradient = self.evaluator.calculate_tension_gradient(self.graph)
        base_beta1 = self.evaluator.calculate_cyclomatic_number(self.graph)

        sandbox_graph = copy.deepcopy(self.graph)
        sandbox_graph.connect(
            proposed_edge.source_id,
            proposed_edge.target_id,
            proposed_edge.tension,
            proposed_edge.resistance
        )

        new_gradient = self.evaluator.calculate_tension_gradient(sandbox_graph)
        new_beta1 = self.evaluator.calculate_cyclomatic_number(sandbox_graph)

        # Principle 1: Core Tension Gradient check
        if new_gradient < self.min_gradient_threshold:
            return False, f"위상 거절: 정체성 코어 장력 구배 파열 (Gradient: {new_gradient:.2f} < {self.min_gradient_threshold})"

        # Principle 2: Cyclomatic topological loop check
        beta1_delta = abs(new_beta1 - base_beta1)
        if beta1_delta > self.max_cyclomatic_delta:
            return False, f"위상 거절: 인과 공간 위상 고리 단절/과팽창 (Cyclomatic Delta: {beta1_delta} > {self.max_cyclomatic_delta})"

        return True, f"위상 승인: 구조 원리 충족 (Gradient: {new_gradient:.2f}, Beta_1 Delta: {beta1_delta})"

    def meta_observe_and_refactor(self) -> List[str]:
        """
        [Meta-Refactoring Loop]
        Scans graph edge accumulated friction. If friction exceeds threshold,
        evaluates refactoring and bypass shortcuts via virtual manifold evaluation.
        """
        logs = []

        for source_id, edge_list in list(self.graph.edges.items()):
            for edge in edge_list:
                if edge.accumulated_friction >= self.friction_threshold_for_refactor:
                    logs.append(
                        f"[메타 관측] 병목 감지: {edge.source_id} -> {edge.target_id} "
                        f"(누적 마찰 손실: {edge.accumulated_friction:.2f})"
                    )

                    # Virtual evaluation of dampening resistance and boosting tension
                    proposed_edge = CausalEdge(
                        edge.source_id,
                        edge.target_id,
                        tension=min(1.0, edge.tension * 1.3),
                        resistance=edge.resistance * 0.3
                    )

                    approved, reason = self.evaluate_relational_refactoring(proposed_edge)

                    if approved:
                        old_resistance = edge.resistance
                        edge.resistance = proposed_edge.resistance
                        edge.tension = proposed_edge.tension
                        logs.append(
                            f"  └─ [지형 재배치 승인] 저항 조정: {old_resistance:.2f} -> {edge.resistance:.2f}, "
                            f"결합강도: {edge.tension:.2f} | {reason}"
                        )

                        # Attempt bypass shortcut creation to subsequent edges
                        target_edges = self.graph.edges.get(edge.target_id, [])
                        for next_edge in target_edges:
                            shortcut_candidate = CausalEdge(
                                edge.source_id,
                                next_edge.target_id,
                                tension=0.85,
                                resistance=0.05
                            )
                            s_approved, s_reason = self.evaluate_relational_refactoring(shortcut_candidate)
                            if s_approved:
                                self.graph.connect(
                                    source_id=edge.source_id,
                                    target_id=next_edge.target_id,
                                    tension=0.85,
                                    resistance=0.05
                                )
                                logs.append(
                                    f"  └─ [인과 우회로 생성] 바이패스 Shortcut 고착화: "
                                    f"{edge.source_id} ==> {next_edge.target_id} | {s_reason}"
                                )
                            else:
                                logs.append(
                                    f"  └─ [인과 우회로 거부] Shortcut 생성 불허: "
                                    f"{edge.source_id} -X-> {next_edge.target_id} | {s_reason}"
                                )
                    else:
                        logs.append(f"  └─ [지형 재배치 거부] {reason}")

                    edge.accumulated_friction = 0.0  # Reset friction after scan

        return logs

    def trigger_self_healing(
        self,
        damaged_path: List[Tuple[str, str]],
        damage_friction_energy: float
    ) -> Tuple[np.ndarray, List[str]]:
        """Triggers plastic self-healing and updates basis vectors."""
        healed_basis, logs = self.healing_engine.absorb_damage_and_reconstruct_basis(
            damaged_path,
            damage_friction_energy,
            self.basis
        )
        self.basis = healed_basis
        return healed_basis, logs
