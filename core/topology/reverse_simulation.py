r"""
Reverse Boundary Value Simulation Engine (역방향 경계값 시뮬레이션 엔진)
=============================================================================
고정된 미래 경계 조건 X_future로부터 현재 X_now로 인과 궤적을 역산하는 내적 시뮬레이션 메커니즘.

원리:
- 목표 미래 상태 X_future를 시스템 부동점(Attractor)으로 고정.
- 현재 상태 X_now에서 X_future로 수렴하기 위한 최소 작용 인과 궤적과
  현재 시점 최적 제어 변인(Control Inputs) 역산.
- O(1) 수준의 역기하학적 피드백 투영.
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class ReverseBoundaryValueSimulator:
    """
    역방향 경계값 시뮬레이션 엔진 (Reverse Boundary Value Simulation Engine)
    """

    def __init__(self, dimension: int = 8, eta: float = 0.1):
        self.dimension = dimension
        self.eta = eta

    def backproject_control(
        self,
        X_now: np.ndarray,
        X_future: np.ndarray,
        scale_axis_S: np.ndarray,
        horizon_steps: int = 5
    ) -> Dict[str, Any]:
        """
        고정된 X_future 경계로부터 현 시점 제어 변인 delta_X_now 역산
        """
        X_n = np.asarray(X_now, dtype=np.float32).reshape(-1)
        X_f = np.asarray(X_future, dtype=np.float32).reshape(-1)
        S = np.asarray(scale_axis_S, dtype=np.float32)

        boundary_gap = X_f - X_n

        S_norm = np.linalg.norm(S) + 1e-8
        required_trajectory = []
        curr_state = X_n.copy()

        for step in range(horizon_steps):
            fraction = (step + 1) / float(horizon_steps)
            target_step_state = X_n + fraction * boundary_gap
            step_friction = target_step_state - curr_state

            # 렌즈 서브스페이스 투영 제어 입력: S @ (S^T @ step_friction) / S_norm
            control_input = np.dot(S, np.dot(S.T, step_friction)) / S_norm
            curr_state = curr_state + self.eta * control_input
            required_trajectory.append(curr_state.copy())

        immediate_control = (required_trajectory[0] - X_n)

        return {
            "immediate_control": immediate_control,
            "boundary_gap_norm": float(np.linalg.norm(boundary_gap)),
            "trajectory": required_trajectory,
            "converged_gap": float(np.linalg.norm(X_f - required_trajectory[-1]))
        }
