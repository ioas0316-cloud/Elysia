r"""
Primitive Cognitive Gate (원시 인지 게이트)
=============================================================================
상위 아키텍처나 사전 정의된 규칙을 완전히 배격하고,
"상태 A와 상태 B가 부딪힐 때 어떤 최소한의 연산 규칙으로 평형 상태의 새로운 관계 노드를 생성하는가?"를
정의하는 단 하나의 미시적 결합식이자 임계 분별기(Threshold Discriminator)입니다.

1. 상태 및 렌즈 투영 방정식 (Lens Projection & State Equations):
   (I, V) = \Phi(X; S)
   - I (Invariant, 불변량): 렌즈 서브스페이스 S 상의 투영 성분
   - V (Variant, 변이량): S에 의해 포획되지 못하고 발생한 '위상 마찰/장력(Phase Friction)'

2. 자기 재정렬 방정식 (Self-Refinement Equation):
   S_{t+1} = S_t - \eta * \nabla_S \mathcal{E}(V)
   - 마찰 V의 장력 에너지를 최소화하는 방향으로 인과 축(렌즈) S를 실시간 업데이트 (O(1) complexity).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class CognitiveGate:
    """
    Primitive Cognitive Gate (원시 인지 게이트)

    데이터와 연산자가 분리되지 않으며, 정보 알갱이 자체가 위상 장력에 의해
    스스로를 인지하고 조율하는 최소 물리 인지 게이트입니다.
    """

    def __init__(
        self,
        dimension: int = 8,
        subspace_dim: Optional[int] = None,
        eta: float = 0.1,
        threshold: float = 0.3,
        max_capacity: float = 2.0,
        sigma_explore: float = 0.05
    ):
        self.dimension = dimension
        self.subspace_dim = subspace_dim if subspace_dim is not None else max(1, dimension // 2)
        self.eta = eta                  # 인과 축 적응률
        self.threshold = threshold      # 스케일 임계선
        self.max_capacity = max_capacity  # 내적 마찰 수용 한계
        self.sigma_explore = sigma_explore # 노이즈 흡수 및 가상 탐색 표준편차

        # 인과 축 렌즈 서브스페이스 S: (dimension, subspace_dim)
        # 초기에는 무작위 정규직교 기저로 설정됩니다.
        Q, _ = np.linalg.qr(np.random.randn(dimension, self.subspace_dim).astype(np.float32))
        self.S = Q

        # 게이트의 누적 마찰 및 상태 기록
        self.last_invariant: Optional[np.ndarray] = None
        self.last_variant: Optional[np.ndarray] = None
        self.accumulated_friction: float = 0.0

    def discriminate(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        분별 연산: (I, V) = \Phi(X; S)
        """
        X_vec = np.asarray(X, dtype=np.float32).reshape(-1)
        if len(X_vec) != self.dimension:
            padded = np.zeros(self.dimension, dtype=np.float32)
            lim = min(len(X_vec), self.dimension)
            padded[:lim] = X_vec[:lim]
            X_vec = padded

        # 서브스페이스 직교 투영: Proj = S @ (S^T @ X)
        proj = np.dot(self.S, np.dot(self.S.T, X_vec))

        mag = np.linalg.norm(proj)
        if mag > self.threshold:
            I = proj
            V = X_vec - proj
        else:
            I = np.zeros_like(X_vec)
            V = X_vec

        self.last_invariant = I
        self.last_variant = V
        return I, V

    def self_refine(self, V: np.ndarray) -> float:
        """
        자기 재정렬 방정식: S_{t+1} = S_t - \eta * \nabla_S \mathcal{E}(V)
        """
        V_vec = np.asarray(V, dtype=np.float32).reshape(-1)
        friction_energy = float(0.5 * np.sum(V_vec ** 2))
        self.accumulated_friction += friction_energy

        # V와 S^T X (또는 S 내 성분) 간의 기울기 회전
        # dS = V outer (S^T X)
        S_proj_coeff = np.dot(self.S.T, V_vec)
        grad_S = np.outer(V_vec, S_proj_coeff)
        self.S -= self.eta * grad_S

        # 직교 정규화 보정
        Q, _ = np.linalg.qr(self.S)
        self.S = Q[:, :self.subspace_dim]

        return friction_energy

    def process(self, X: np.ndarray) -> Dict[str, Any]:
        """
        단일 게이트 통섭 프로세스
        """
        I, V = self.discriminate(X)
        friction_energy = self.self_refine(V)

        action_triggered = False
        action_reprojected = None
        if friction_energy > self.max_capacity:
            action_triggered = True
            action_reprojected = -np.dot(self.S, np.dot(self.S.T, V))

        return {
            "invariant": I,
            "variant": V,
            "friction_energy": friction_energy,
            "scale_axis": self.S.copy(),
            "action_triggered": action_triggered,
            "action_reprojected": action_reprojected
        }

    def simulate(self, hypothetical_X: np.ndarray, steps: int = 5) -> Dict[str, Any]:
        """
        내부 가상 시뮬레이션 (Prediction / Imagination)
        """
        S_virtual = self.S.copy()
        X_curr = hypothetical_X.copy()
        trajectory = []

        for _ in range(steps):
            proj = np.dot(S_virtual, np.dot(S_virtual.T, X_curr))
            V_virt = X_curr - proj
            friction = float(0.5 * np.sum(V_virt ** 2))

            S_proj_coeff = np.dot(S_virtual.T, V_virt)
            grad_S = np.outer(V_virt, S_proj_coeff)
            S_virtual -= self.eta * grad_S
            Q, _ = np.linalg.qr(S_virtual)
            S_virtual = Q[:, :self.subspace_dim]

            trajectory.append({
                "friction": friction,
                "projected_mag": float(np.linalg.norm(proj))
            })

            X_curr = proj

        return {
            "initial_friction": trajectory[0]["friction"],
            "final_friction": trajectory[-1]["friction"],
            "trajectory": trajectory,
            "converged": trajectory[-1]["friction"] <= trajectory[0]["friction"]
        }


class RecursiveCognitiveStack:
    """
    재귀적 인지 게이트 중첩체 (Recursive Cognitive Gate Stack)
    """

    def __init__(self, layers: int = 3, dimension: int = 8, eta: float = 0.1):
        self.layers = [
            CognitiveGate(dimension=dimension, eta=eta, threshold=0.1 * (l + 1))
            for l in range(layers)
        ]

    def process_hierarchical(self, raw_signal: np.ndarray) -> Dict[str, Any]:
        current_input = raw_signal
        layer_outputs = []

        for level, gate in enumerate(self.layers):
            res = gate.process(current_input)
            layer_outputs.append({
                "level": level + 1,
                "invariant": res["invariant"],
                "variant": res["variant"],
                "friction": res["friction_energy"],
                "action_triggered": res["action_triggered"]
            })
            current_input = res["invariant"]

        top_attractor = current_input
        return {
            "layer_outputs": layer_outputs,
            "top_attractor": top_attractor,
            "total_friction": sum(l["friction"] for l in layer_outputs)
        }
