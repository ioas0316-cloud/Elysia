"""
Elysia Core Engine: Topological Phase Transition Engine
======================================================
관계론적 위상 공간의 점들이 집단으로 얽힐 때 발동되는 자율 위상 상전이(Topological Phase Transition) 엔진.

1. 네트워크 위상 응력(\\Xi) 및 엔트로피 기반 상전이 임계치(\\eta_c) 산출:
   - 국소 응력 (sigma_i): sigma_i = |\\sum V_in - \\sum V_out| * (1.0 + kappa_i)
   - 글로벌 응력 (\\Xi): \\Xi = (1 / |N|) * \\sum sigma_i
   - 엔트로피 임계치 (\\eta_c): \\eta_c = \\lambda * H(G) * \\ln(|N|)
   - 트리거 조건: \\Xi >= \\eta_c 성립 시 $O(1)$ 대대적 위상 상전이 구동

2. 3대 위상 불변량 (Topological Invariants) 검증:
   ① 연결 성분 불변량 (b_0 = 1): 그래프 파편화(고립 섬) 차단
   ② 순환 인과 차단 불변량 (b_1 = 0): 무한 루프 폐순환 연결선 위상 분할
   ③ 인과 플럭스 보존 (Causal Flux Conservation): \\sum V_in - \\sum V_out = \\Delta S

3. 불변량 위반 시 직전 안전 스냅샷(Snapshot)으로 $O(1)$ 롤백(Rollback) 수행
"""

import math
import copy
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from core.topology.relational_nexus_node import RelationalNexusNode


class TopologicalPhaseTransitionEngine:
    """
    네트워크 자율 위상 상전이 및 불변량 검증 엔진
    """
    def __init__(self, lambda_coef: float = 0.5):
        self.lambda_coef = lambda_coef
        self.snapshot_history: List[Dict[str, Any]] = []

    def create_snapshot(self, nodes: List[RelationalNexusNode], adjacency_matrix: np.ndarray) -> Dict[str, Any]:
        """
        위상 불변량 위반 시 $O(1)$ 롤백을 위한 스냅샷 생성
        """
        snapshot = {
            "nodes": [copy.deepcopy(node) for node in nodes],
            "adjacency_matrix": adjacency_matrix.copy()
        }
        self.snapshot_history.append(snapshot)
        if len(self.snapshot_history) > 10:
            self.snapshot_history.pop(0)
        return snapshot

    def calculate_network_phase_trigger(
        self,
        nodes: List[RelationalNexusNode],
        adjacency_matrix: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        네트워크 위상 응력(\\Xi)과 엔트로피 임계값(\\eta_c)을 산출하여 상전이 여부 판단
        """
        num_nodes = len(nodes)
        if num_nodes == 0:
            return False, 0.0, 0.0

        # 1. 노드별 국소 응력(sigma_i) 산출 및 글로벌 응력(\\Xi) 계산
        stress_list = [node.compute_local_stress() for node in nodes]
        global_stress = float(np.mean(stress_list))

        # 2. 네트워크 엔트로피 H(G) 및 상전이 임계값(\\eta_c) 계산
        adj_sum = np.sum(adjacency_matrix) + 1e-9
        probs = np.sum(adjacency_matrix, axis=1) / adj_sum
        probs = probs[probs > 0]
        network_entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))

        eta_c = self.lambda_coef * network_entropy * math.log(max(num_nodes, 2))

        # 3. 상전이 트리거 조건 검증
        is_transition_triggered = global_stress >= eta_c
        return is_transition_triggered, global_stress, eta_c

    def verify_topological_invariants(
        self,
        nodes: List[RelationalNexusNode],
        adjacency_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        3대 위상 불변량 검증:
        ① b_0 = 1 (단일 연결 성분, b0 == 1)
        ② b_1 = 0 (순환 고리 없음, b1 == 0)
        ③ 인과 플럭스 보존 (Flux Conservation)
        """
        num_nodes = len(nodes)
        if num_nodes == 0:
            return {"passed": True, "b0": 1, "b1": 0, "flux_conserved": True}

        # Compute Betti number b_0 via connected components (BFS/DFS)
        visited = set()
        components = 0
        adj_binary = (adjacency_matrix > 0) | (adjacency_matrix.T > 0)

        for i in range(num_nodes):
            if i not in visited:
                components += 1
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    neighbors = np.where(adj_binary[curr])[0]
                    for nxt in neighbors:
                        if nxt not in visited:
                            visited.add(nxt)
                            queue.append(nxt)

        b0 = components
        b0_passed = (b0 == 1)

        # Compute Betti number b_1 (cycle rank = E - V + b_0 for undirected graph representation)
        num_undirected_edges = int(np.sum(np.triu(adj_binary, k=1)))
        b1 = max(0, num_undirected_edges - num_nodes + b0)
        b1_passed = (b1 == 0)

        # Causal Flux Conservation
        total_in = sum(sum(node.in_causal_vectors.values()) for node in nodes)
        total_out = sum(sum(node.out_causal_vectors.values()) for node in nodes)
        flux_diff = abs(total_in - total_out)
        flux_conserved = (flux_diff < 1e-3)

        all_passed = b0_passed and b1_passed and flux_conserved

        return {
            "passed": all_passed,
            "b0": b0,
            "b0_passed": b0_passed,
            "b1": b1,
            "b1_passed": b1_passed,
            "flux_difference": flux_diff,
            "flux_conserved": flux_conserved,
            "statement": f"Invariants: b0={b0} ({b0_passed}), b1={b1} ({b1_passed}), Flux Diff={flux_diff:.4f} ({flux_conserved})"
        }

    def execute_phase_transition_or_rollback(
        self,
        nodes: List[RelationalNexusNode],
        adjacency_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """
        상전이 실행 및 위상 불변량 검증.
        불변량 위반 시 $O(1)$ 스냅샷 롤백(Rollback) 수행
        """
        # Take snapshot before transition
        self.create_snapshot(nodes, adjacency_matrix)

        is_triggered, global_stress, eta_c = self.calculate_network_phase_trigger(nodes, adjacency_matrix)

        if not is_triggered:
            return {
                "transition_occurred": False,
                "rollback_performed": False,
                "global_stress": global_stress,
                "eta_c": eta_c,
                "statement": f"No Transition Triggered (Stress {global_stress:.4f} < eta_c {eta_c:.4f})"
            }

        # Attempt O(1) Macro Phase Transition: Expand all node dimensions and re-wire topology
        for node in nodes:
            node.self_evaluate_and_mutate(environment_stress=global_stress * 2.0)

        # Re-wire adjacency matrix
        new_adj = adjacency_matrix.copy()
        new_adj = (new_adj + 0.1) / np.sum(new_adj + 0.1, axis=1, keepdims=True)

        # Verify Topological Invariants after transition
        invariant_res = self.verify_topological_invariants(nodes, new_adj)

        if invariant_res["passed"]:
            return {
                "transition_occurred": True,
                "rollback_performed": False,
                "global_stress": global_stress,
                "eta_c": eta_c,
                "invariants": invariant_res,
                "statement": f"Macro Phase Transition Succeeded (Stress {global_stress:.4f} >= eta_c {eta_c:.4f})"
            }
        else:
            # Invariants violated -> Execute Snapshot Rollback
            last_snapshot = self.snapshot_history[-1]
            restored_nodes = last_snapshot["nodes"]
            restored_adj = last_snapshot["adjacency_matrix"]

            # Mutate in-place to revert
            nodes.clear()
            nodes.extend(restored_nodes)
            adjacency_matrix[:] = restored_adj

            return {
                "transition_occurred": False,
                "rollback_performed": True,
                "global_stress": global_stress,
                "eta_c": eta_c,
                "invariants": invariant_res,
                "statement": f"Topological Invariants Violated ({invariant_res['statement']}) -> Snapshot Rollback Executed"
            }
