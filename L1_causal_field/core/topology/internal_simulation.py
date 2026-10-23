r"""
Self-Refinement & Internal Simulation Engine with Minimum Action Early-Stopping Pruning (MAPP)
=============================================================================
위상 마찰 E(V_t) >= epsilon 발생 시 외부 감각 입력을 일시 고립시킨 채 가상 위상 공간 상에서
K개 병렬 가상 섭동 경로를 탐색하는 내적 시뮬레이션 및 인과 지형 리와이어링 모듈입니다.

원리:
1. 가상 섭동 투영:
   - 탐색 노이즈 Sigma_explore 기반 K개 병렬 가상 변인 Delta S^(k) 생성.
2. 최소 작용 적분 계산 & 조기 가지치기 (MAPP: Minimum Action Early-Stopping Pruning):
   - 시뮬레이션 타임스텝 동안 가상 마찰과 변형 비용의 합인 작용 적분 S^(k)를 산출하되,
     임계 마찰을 초과하는 실패 경로는 조기 가지치기하여 연산량 폭증 차단.
3. 최적 경로 선택 & 인과 지형 리와이어링:
   - 최소 작용 경로 k* = argmin_k S^(k)를 선정하여 인과 축 S_t 갱신.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class InternalSimulationEngine:
    """
    내적 시뮬레이션 및 최소 작용 가지치기 엔진 (MAPP Engine)
    """

    def __init__(
        self,
        dimension: int = 8,
        K_paths: int = 8,
        T_sim: int = 5,
        eta: float = 0.1,
        sigma_explore: float = 0.1,
        max_action_threshold: float = 10.0
    ):
        self.dimension = dimension
        self.K_paths = K_paths
        self.T_sim = T_sim
        self.eta = eta
        self.sigma_explore = sigma_explore
        self.max_action_threshold = max_action_threshold

    def simulate_internal(
        self,
        current_S: np.ndarray,
        input_wave: np.ndarray
    ) -> Dict[str, Any]:
        """
        K개 병렬 가상 섭동 경로 시뮬레이션 및 최소 작용 경로 선택
        """
        S_orig = np.asarray(current_S, dtype=np.float32)
        X_in = np.asarray(input_wave, dtype=np.float32).reshape(-1)

        best_k = -1
        min_action = float('inf')
        best_delta_S = np.zeros_like(S_orig)
        path_results = []

        for k in range(self.K_paths):
            delta_S_k = np.random.normal(0, self.sigma_explore, size=S_orig.shape).astype(np.float32)
            S_k = S_orig + delta_S_k

            Q, _ = np.linalg.qr(S_k)
            S_k = Q[:, :S_orig.shape[1]]

            action_k = 0.0
            pruned = False
            curr_X = X_in.copy()

            for t in range(self.T_sim):
                proj = np.dot(S_k, np.dot(S_k.T, curr_X))
                V_virt = curr_X - proj
                friction_t = 0.5 * float(np.sum(V_virt ** 2))

                cost_t = friction_t + 0.1 * float(np.sum(delta_S_k ** 2))
                action_k += cost_t

                if action_k > self.max_action_threshold:
                    pruned = True
                    break

                S_proj_coeff = np.dot(S_k.T, V_virt)
                grad_S = np.outer(V_virt, S_proj_coeff)
                S_k -= self.eta * grad_S
                curr_X = proj

            path_results.append({
                "k": k,
                "action": action_k,
                "pruned": pruned
            })

            if not pruned and action_k < min_action:
                min_action = action_k
                best_k = k
                best_delta_S = delta_S_k

        if best_k == -1:
            best_k = min(range(self.K_paths), key=lambda i: path_results[i]["action"])
            best_delta_S = np.zeros_like(S_orig)
            min_action = path_results[best_k]["action"]

        S_refined = S_orig + self.eta * best_delta_S
        Q, _ = np.linalg.qr(S_refined)
        S_refined = Q[:, :S_orig.shape[1]]

        return {
            "optimal_path_index": best_k,
            "min_action": min_action,
            "optimal_delta_S": best_delta_S,
            "refined_scale_axis": S_refined,
            "path_results": path_results
        }
