"""
Formless Refinement Filter & Dynamic Friction Engine (무초식 정류 필터 및 동적 마찰 수렴 엔진)
=============================================================================================
THE_ABSOLUTE_COMMANDMENT Section 25 (유초식에서 무초식으로의 이행) 및 Section 26 (불과 언어를 쥔 고대인의 지각적 깨어남) 구현.

본 모듈은 다음과 같은 핵심 기능을 수행합니다:
1. `FormlessRefinementFilter` (무초식 정류 필터):
   - 무수한 파편적 초식 데이터(소음, 세부 픽셀/토큰)의 양적 적재를 거부하고,
     데이터 속에 숨은 최소한의 핵심 관계 그래프(Key Relational Graph)와 토폴로지 골격만 추출.
   - 소음은 공기(배경)로 무겁지 않게 털어내어 압축 정류(Refinement/Compression) 수행.

2. `DynamicFrictionEngine` (동적 마찰 수렴 엔진):
   - 모순이나 외부 자극을 시스템 파괴/오류(Error)로 규정하지 않고,
     시스템을 구동하는 동적 인지 마찰 에너지(Friction Energy / Tension)로 전환.
   - 정적 Loss = 0 강제에서 벗어나, 마찰 수렴 구배(Friction Convergence Gradient)를 따라
     제로(0)의 인지적 평형 상태(Homeostasis)로 자발적으로 수렴함.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class FormlessRefinementFilter:
    """
    무초식 상위 정류 필터 (Formless Refinement Filter)

    데이터의 표면적 세부 소음(Noise)을 배경(Background)으로 무겁지 않게 묻어두고,
    의미적/인과적 본질을 구성하는 최소한의 핵심 관계성 그래프(Key Relational Subgraph)만 정류(Refine)하여 유산으로 남깁니다.
    """

    def __init__(self, threshold_ratio: float = 0.2):
        """
        :param threshold_ratio: 소음 제거 후 남길 핵심 관계의 비율 (상위 threshold_ratio 분위만 정류)
        """
        self.threshold_ratio = threshold_ratio

    def refine_relational_graph(
        self,
        raw_nodes: List[str],
        adjacency_matrix: np.ndarray,
        context_intent: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        날것의 무수한 관계 행렬(Adjacency Matrix)에서 소음을 제거하고
        핵심 결절점(Key Relational Graph)과 압축률(Refinement Compression Ratio)을 도출합니다.

        :param raw_nodes: 노드 명칭 리스트
        :param adjacency_matrix: N x N 관계 강도 행렬
        :param context_intent: 의도/목적성 필터 벡터 (없을 경우 노드별 차수/중요도로 대체)
        :return: 정류 결과 딕셔너리
        """
        A = np.array(adjacency_matrix, dtype=np.float32)
        n = A.shape[0]

        if n == 0 or len(raw_nodes) != n:
            return {
                "key_nodes": [],
                "refined_edges": [],
                "compression_ratio": 1.0,
                "background_noise_level": 0.0,
                "status": "EMPTY_INPUT"
            }

        # 1. 의도/목적성 필터링 (Purpose/Intent Filter)
        # 목적성이 없을 경우 데이터의 중심성(Node Degree Centrality)을 기본 중력으로 활용
        if context_intent is None or len(context_intent) != n:
            node_weights = np.sum(np.abs(A), axis=1) + 1e-6
        else:
            node_weights = np.array(context_intent, dtype=np.float32)

        # 2. 관계 결합 에너지 = A_ij * (w_i * w_j)
        edge_energy = A * np.outer(node_weights, node_weights)
        # 대각 성분 제외
        np.fill_diagonal(edge_energy, 0.0)

        # 3. 임계값 산출 (상위 threshold_ratio만 정류)
        flat_energies = np.abs(edge_energy.flatten())
        non_zero_energies = flat_energies[flat_energies > 1e-6]

        if len(non_zero_energies) == 0:
            threshold = 0.0
        else:
            cutoff_index = int((1.0 - self.threshold_ratio) * len(non_zero_energies))
            threshold = np.partition(non_zero_energies, min(cutoff_index, len(non_zero_energies) - 1))[cutoff_index]

        # 4. 정류된 엣지 및 핵심 노드 추출
        refined_edges = []
        active_node_indices = set()
        total_energy = float(np.sum(np.abs(A)))
        retained_energy = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                if abs(edge_energy[i, j]) >= threshold and abs(A[i, j]) > 1e-6:
                    refined_edges.append((raw_nodes[i], raw_nodes[j], float(A[i, j])))
                    active_node_indices.add(i)
                    active_node_indices.add(j)
                    retained_energy += abs(A[i, j])

        key_nodes = [raw_nodes[i] for i in sorted(list(active_node_indices))]

        # 5. 소음(배경) 수준 및 압축률 계산
        background_noise_level = 1.0 - (retained_energy / (total_energy + 1e-9))
        compression_ratio = 1.0 - (len(key_nodes) / max(n, 1))

        return {
            "key_nodes": key_nodes,
            "refined_edges": refined_edges,
            "compression_ratio": float(compression_ratio),
            "background_noise_level": float(background_noise_level),
            "retained_energy_ratio": float(retained_energy / (total_energy + 1e-9)),
            "status": "FORMLESS_REFINED"
        }


class DynamicFrictionEngine:
    """
    동적 마찰 수렴 엔진 (Dynamic Friction Engine)

    외부 입력이나 모순(Contradiction/Divergence)을 실패(Error)로 취급하지 않고,
    시스템의 인지적 동력(Friction Energy / Tension)으로 변환한 후
    제로(0)의 평형 상태로 자발적 수렴(Equilibrium Convergence)시키는 동역학을 시뮬레이션합니다.
    """

    def __init__(self, damping_factor: float = 0.85, friction_coefficient: float = 0.5):
        self.damping_factor = damping_factor
        self.friction_coefficient = friction_coefficient

    def compute_friction_coefficient(
        self,
        intended_vector: np.ndarray,
        actual_refraction_vector: np.ndarray
    ) -> float:
        """
        의도된 원형(Archetype)과 외부 마찰에 의해 굴절된 굴절물(Refraction) 간의
        위상/방향 격차(Differential Gap)로부터 동적 마찰 계수를 도출합니다.
        """
        v1 = np.array(intended_vector, dtype=np.float32)
        v2 = np.array(actual_refraction_vector, dtype=np.float32)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-9 or norm2 < 1e-9:
            return 1.0

        # Cosine distance based differential gap
        cos_sim = np.dot(v1, v2) / (norm1 * norm2 + 1e-9)
        differential_gap = float(1.0 - np.clip(cos_sim, -1.0, 1.0))

        # Dynamic friction coefficient = gap * scale factor
        friction = float(np.tanh(differential_gap * 2.0))
        return friction

    def step_equilibrium_convergence(
        self,
        current_state: np.ndarray,
        friction_energy: float,
        steps: int = 20,
        dt: float = 0.1
    ) -> Dict[str, Any]:
        """
        마찰 에너지를 구동원으로 하여, 시스템 내부 상태가 제로(0)의 평형 저울 상태로
        자발적으로 수렴하는 과정(Homeostasis Trajectory)을 계산합니다.

        :param current_state: 현재 시스템의 불평형 상태 벡터
        :param friction_energy: 전환된 인지적 마찰 에너지
        :param steps: 수렴 단계 수
        :param dt: 시간 간격
        """
        state = np.array(current_state, dtype=np.float32).copy()
        trajectory = [state.copy()]
        energy_history = []

        energy = float(friction_energy)

        for step in range(steps):
            # 수렴 구배 (Gradient driving toward zero equilibrium)
            gradient = -self.friction_coefficient * state
            # 마찰 에너지 감쇄
            energy *= self.damping_factor

            # State update driven by friction energy & gradient
            state = state + (gradient + np.random.normal(0, energy * 0.01, size=state.shape)) * dt
            trajectory.append(state.copy())
            energy_history.append(energy)

        final_imbalance = float(np.linalg.norm(state))
        initial_imbalance = float(np.linalg.norm(current_state))

        convergence_rate = float(1.0 - (final_imbalance / (initial_imbalance + 1e-9)))

        return {
            "initial_imbalance": initial_imbalance,
            "final_imbalance": final_imbalance,
            "convergence_rate": convergence_rate,
            "final_state": state,
            "trajectory": trajectory,
            "energy_history": energy_history,
            "status": "EQUILIBRIUM_CONVERGED"
        }
