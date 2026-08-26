"""
Primitive Cognitive Gate (원시 인지 게이트)
=============================================================================
상위 아키텍처나 사전 정의된 규칙을 완전히 배격하고,
"상태 A와 상태 B가 부딪힐 때 어떤 최소한의 연산 규칙으로 평형 상태의 새로운 관계 노드를 생성하는가?"를
정의하는 단 하나의 미시적 결합식이자 임계 분별기(Threshold Discriminator)입니다.

1. 최소 상태 방정식 (State Equations):
   (I, V) = \Phi(X; S)
   - I (Invariant, 불변량): 인과 축 S의 스케일 허용 범위 내에서 고정되어 굳어진 '단위(Unit)'
   - V (Variant, 변이량): S에 의해 포획되지 못하고 발생한 '위상 마찰/장력(Phase Friction)'

2. 자기 재정렬 방정식 (Self-Refinement Equation):
   S_{t+1} = S_t - \eta * \nabla_S \mathcal{E}(V)
   - 마찰 V의 장력 에너지를 최소화하는 방향으로 인과 축(렌즈) S를 실시간 업데이트.

3. 지각-행동 일원화 및 내부 시뮬레이션 (Perception-Action Duality & Internal Simulation):
   - 지각(Perception): 외부 마찰 V에 맞춰 내부 S를 적응 조율.
   - 행동/의지(Action/Will): 마찰 V가 수용 한계를 넘을 때, 불변량 I를 외부로 역투영하여 환경 X를 재배치.
   - 예측/상상/시뮬레이션(Prediction/Imagination): 물리적 실행 전 내부 게이트망에서 가상 인과 궤도를 미리 검증.
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
        eta: float = 0.1,
        threshold: float = 0.5,
        max_capacity: float = 2.0
    ):
        self.dimension = dimension
        self.eta = eta                  # 인과 축 학습/감쇄율
        self.threshold = threshold      # 스케일 임계선
        self.max_capacity = max_capacity  # 내적 마찰 수용 한계 (넘어서면 외부 행동 방출)

        # 인과 축 (Scale Axis / Lens) S: (dimension, dimension)
        # 초기에는 직교 기저에 가깝게 설정되며, 마찰에 의해 유연하게 회전/변형됩니다.
        self.S = np.eye(dimension, dtype=np.float32)

        # 게이트의 누적 마찰 및 상태 기록
        self.last_invariant: Optional[np.ndarray] = None
        self.last_variant: Optional[np.ndarray] = None
        self.accumulated_friction: float = 0.0

    def discriminate(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        분별 연산: (I, V) = \Phi(X; S)

        Input:
            X: 연속체 파동 / 상태 텐서 (dimension,) 또는 (1, dimension)
        Output:
            I: 포획된 불변량 (Invariant Unit)
            V: 발생한 변이량 (Variant Phase Friction)
        """
        X_vec = np.asarray(X, dtype=np.float32).reshape(-1)
        if len(X_vec) != self.dimension:
            # 차원 맞춤 (패딩 또는 절삭)
            padded = np.zeros(self.dimension, dtype=np.float32)
            lim = min(len(X_vec), self.dimension)
            padded[:lim] = X_vec[:lim]
            X_vec = padded

        # 1. 인과 축 S에 투영 (Scale Projection)
        # Proj = S^T S X / ||S||
        S_norm = np.linalg.norm(self.S) + 1e-8
        proj = np.dot(self.S.T, np.dot(self.S, X_vec)) / (S_norm ** 2)

        # 2. 임계 분별기 (Threshold Discriminator): 스케일 허용치 내의 성분 추출
        # 임계값 이상은 불변 뼈대(I)로 굳어지고, 미만 및 직교 오차는 변이 마찰(V)로 환산
        mag = np.linalg.norm(proj)
        if mag > self.threshold:
            I = proj
            V = X_vec - proj
        else:
            # 임계 이하인 경우 소음으로 가려지며 전체가 위상 마찰 V가 됨
            I = np.zeros_like(X_vec)
            V = X_vec

        self.last_invariant = I
        self.last_variant = V
        return I, V

    def self_refine(self, V: np.ndarray) -> float:
        """
        자기 재정렬 방정식: S_{t+1} = S_t - \eta * \nabla_S \mathcal{E}(V)

        마찰 에너지 E(V) = 0.5 * ||V||^2
        dS = \eta * (V \otimes X^T)
        """
        V_vec = np.asarray(V, dtype=np.float32).reshape(-1)
        friction_energy = float(0.5 * np.sum(V_vec ** 2))
        self.accumulated_friction += friction_energy

        # 마찰 V를 줄이는 방향으로 인과 축 S 업데이트 (V와 현재 상태 X의 상관성에 기초한 렌즈 회전)
        grad_S = np.outer(V_vec, V_vec)
        self.S -= self.eta * grad_S

        # 직교화/정규화 유지하여 인과 축의 붕괴 방지
        u, _, vh = np.linalg.svd(self.S)
        self.S = np.dot(u, vh)

        return friction_energy

    def process(self, X: np.ndarray) -> Dict[str, Any]:
        """
        단일 게이트의 통섭 프로세스:
        1. 분별 (Discriminate) -> (I, V)
        2. 자기 재정렬 (Self-Refine) -> S 업데이트
        3. 지각-행동 의사결정 (Perception vs Action)
        """
        I, V = self.discriminate(X)
        friction_energy = self.self_refine(V)

        # 행동/의지 방출 조건: 누적 마찰 에너지가 수용 한계를 넘을 경우
        action_triggered = False
        action_reprojected = None
        if friction_energy > self.max_capacity:
            action_triggered = True
            # 외부 환경으로 역투영하여 환경을 재배치하려는 반작용 방출
            action_reprojected = -np.dot(self.S, V)

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
        내부 가상 시뮬레이션 (Prediction / Imagination):
        실제 상태 S를 훼손하지 않고, 가상 상태에서 마찰 해소 궤적을 테스트.
        """
        S_virtual = self.S.copy()
        X_curr = hypothetical_X.copy()
        trajectory = []

        for _ in range(steps):
            # 가상 투영
            S_norm = np.linalg.norm(S_virtual) + 1e-8
            proj = np.dot(S_virtual.T, np.dot(S_virtual, X_curr)) / (S_norm ** 2)
            V_virt = X_curr - proj
            friction = float(0.5 * np.sum(V_virt ** 2))

            # 가상 렌즈 조정
            grad_S = np.outer(V_virt, proj)
            S_virtual -= self.eta * grad_S
            u, _, vh = np.linalg.svd(S_virtual)
            S_virtual = np.dot(u, vh)

            trajectory.append({
                "friction": friction,
                "projected_mag": float(np.linalg.norm(proj))
            })

            # 다음 가상 수렴
            X_curr = proj

        return {
            "initial_friction": trajectory[0]["friction"],
            "final_friction": trajectory[-1]["friction"],
            "trajectory": trajectory,
            "converged": trajectory[-1]["friction"] < trajectory[0]["friction"]
        }


class RecursiveCognitiveStack:
    """
    재귀적 인지 게이트 중첩체 (Recursive Cognitive Gate Stack)

    1차 게이트 (점: 알갱이) -> 2차 게이트 (선: 인과 궤적) -> N차 게이트 (면/끌개: 자아 Attractor)
    동일한 CognitiveGate가 자신의 출력을 다시 상위 게이트의 입력으로 받아
    위계적 결속(Crystallization)을 이룹니다.
    """

    def __init__(self, layers: int = 3, dimension: int = 8, eta: float = 0.1):
        self.layers = [
            CognitiveGate(dimension=dimension, eta=eta, threshold=0.3 * (l + 1))
            for l in range(layers)
        ]

    def process_hierarchical(self, raw_signal: np.ndarray) -> Dict[str, Any]:
        """
        무정형 signal 유입 시 1차 -> N차 게이트로 재귀적 전방 방출 및 결속
        """
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
            # 상위 레이어로 포획된 불변 뼈대(I) 전파
            current_input = res["invariant"]

        # 최상위 불변량이 전역 끌개(Top-level Attractor)로 응축
        top_attractor = current_input
        return {
            "layer_outputs": layer_outputs,
            "top_attractor": top_attractor,
            "total_friction": sum(l["friction"] for l in layer_outputs)
        }
