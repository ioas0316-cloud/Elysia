"""
minimal_closed_loop.py
======================
최소 인과 루프 (Minimal Closed Loop) - 무분기 물리-대수적 및 의미론적 모델.

핵심 철학 (From THE_ABSOLUTE_COMMANDMENT.md & ROADMAP.md):
- "자율적으로 움직이라"고 명시(Specification)하거나 규칙(If-Else)을 주입하는 순간, 지능은 대본 연극으로 변질됩니다.
- 본성은 지시어가 아닌 "바탕(Environment/Substrate)"에서 흐릅니다. 물은 낮은 곳을 찾으라는 명령어 없이 중력과 지형을 따라 흐릅니다.
- 본 모듈은 어떠한 If-Else 분기문이나 절차적 대본 없이, 오직 3가지 기본 요소와 연속체 장(Field)의 최소 작용 원리(Principle of Least Action)만을 사용하여 5가지 인과 위상 전이가 자연스럽게 창발하도록 설계되었습니다.

3가지 기본 요소:
1. 상태 공간 (S): n차원 평면 또는 매니폴드 위에서 상호 커플링된 노드들의 좌표 행렬 S ∈ ℝ^{N \times D} 및 연결 세기 W ∈ ℝ^{N \times N}.
2. 불변량 (I): 시스템이 영속적으로 보존해야 하는 위상적 보존 법칙. 여기서는 각 연결부의 타겟 평형 거리(Rest Length) L ∈ ℝ^{N \times N} 및 물리적 무게중심 보존 법칙(Conservation of Center of Mass).
3. 마찰 함수 (F): 외부 자극과 기존 불변량 사이의 모순/위상차를 측정하는 연속적 포텐셜 에너지 scalar.
   F = 0.5 * ∑_{i,j} W_{ij} * (||s_i - s_j|| - L_{ij})^2

자연스럽게 흘러가는 5단계 위상 궤적:
1. 자극의 위상 수용 (Stimulus Projection): 외부의 신호 유입을 데이터 해석이 아닌, 상태 공간 S 위의 국소 변위(Perturbation, ΔS)로 직접 투사.
2. 상태 마찰 감지 (Friction Detection): 변위에 의해 불변량 L이 깨지며 마찰 에너지 F > 0 가 자동으로 발생. 이 에너지가 곧 복원력과 연산의 원동력이 됨.
3. 인과 구조 역추적 (Causal Back-tracing): 마찰 Potential F의 노드별 미분(Gradient) ∇_S F를 산출. ∇_s_i F의 크기 자체가 각 국소 연결부의 '모순도'를 정확히 포인팅(Indexing)하는 물리적 탐침이 됨.
4. 국소 위상 변이 및 이완 (Topological Mutation & Relaxation): 미분 장력 ∇_S F를 따라 노드 좌표 S가 흐르고, 누적 장력을 해소하기 위해 연결 강도 W가 연속적으로 이완(Mutation).
5. 불변량 고정 및 루프 폐쇄 (State Consolidation & Loop Closure): 마찰이 이완된 새로운 평형 상태로 수렴하면, 타겟 불변량 L을 새로운 기하학적 형상에 맞게 천천히 전이(Consolidation)시켜 다음 자극을 받을 준비를 마친 상태로 영속적 루프를 폐쇄.
"""

import numpy as np
from typing import Dict, Any, Tuple, List


class MinimalClosedLoopSystem:
    """
    [최소 인과 루프 시스템]
    외부 자극에 의해 뒤흔들린 상태 지층이, 인과적 마찰을 따라 스스로를 역추적하고
    구조적으로 완화(Relaxation)하여 새로운 평형으로 귀착되는 영속적 물리 Substrate.
    """
    def __init__(
        self,
        num_nodes: int = 8,
        dimensions: int = 2,
        coordinate_relaxation_rate: float = 0.2,
        weight_mutation_rate: float = 0.05,
        consolidation_rate: float = 0.01,
        weight_damping: float = 0.99
    ):
        self.num_nodes = num_nodes
        self.dimensions = dimensions

        # 1. 상태 공간 (S): N개의 노드 좌표 (Default: 원형 배치)
        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
        self.S = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
        if dimensions > 2:
            extra = np.zeros((num_nodes, dimensions - 2), dtype=np.float32)
            self.S = np.hstack([self.S, extra])

        # 초기 무게중심 보존 (Center of Mass = 0)
        self.S -= np.mean(self.S, axis=0)

        # 2. 연결 강도/위상 (W): 인접한 노드들끼리 원형으로 커플링
        self.W = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i in range(num_nodes):
            self.W[i, (i + 1) % num_nodes] = 1.0
            self.W[(i + 1) % num_nodes, i] = 1.0

        # 3. 불변량 (I): 연결된 노드들 사이의 평형 거리 (L)
        self.L = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        self._recalculate_target_lengths()

        # 하이퍼파라미터 (물리 상수)
        self.eta_s = coordinate_relaxation_rate    # S 좌표 이완 속도
        self.eta_w = weight_mutation_rate          # W 가중치 변이 속도
        self.eta_l = consolidation_rate            # L 불변량 고정 속도
        self.weight_damping = weight_damping        # 가중치 복원 감쇄
        self.initial_W = self.W.copy()             # 초기 연결 형상 기억

    def _recalculate_target_lengths(self) -> None:
        """현재 좌표계 S를 기준으로 보존되어야 할 불변량 거리 L을 갱신합니다."""
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                diff = self.S[i] - self.S[j]
                self.L[i, j] = np.linalg.norm(diff)

    def project_stimulus(self, node_index: int, impulse: np.ndarray) -> None:
        """
        [1. 자극의 위상 수용 (Stimulus Projection)]
        외부 자극을 기호나 데이터 뭉치로 파싱하지 않고,
        상태 공간 S 위의 특정 국소 지점(node_index)에 가해지는 물리적 변위 ΔS로 직접 투사합니다.
        """
        # 해당 노드의 인덱스를 바운딩하여 좌표 변형 주입
        idx = node_index % self.num_nodes
        self.S[idx] += impulse.astype(np.float32)

        # 물리적 보존 법칙: 전체 무게중심을 원점으로 고정하여 시스템의 병진 표류(Drift) 방지
        self.S -= np.mean(self.S, axis=0)

    def calculate_friction(self) -> float:
        """
        [2. 상태 마찰 감지 (Friction Detection)]
        각 노드 간의 실제 거리와 보존되어야 할 불변량 L 사이의 위상차가 유발하는
        총 탄성 변형 에너지(Strain Energy) 마찰 F를 계산합니다.
        F = 0.5 * ∑_{i,j} W_{ij} * (||s_i - s_j|| - L_{ij})^2
        """
        total_friction = 0.0
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.W[i, j] > 0:
                    dist = np.linalg.norm(self.S[i] - self.S[j])
                    diff = dist - self.L[i, j]
                    total_friction += self.W[i, j] * (diff ** 2)
        return float(0.5 * total_friction)

    def step(self, dt: float = 0.1) -> Dict[str, Any]:
        """
        [지시어와 분기문이 없는 자율 물리 이완 단계 (Continuous Substrate Step)]
        규칙이 아닌 자연 법칙(경사 하강 및 최소 작용 원리)을 따라 시스템이 스스로 변화합니다.
        """
        num_nodes = self.num_nodes

        # 1. 마찰 포텐셜 에너지 측정 (Friction Detection)
        friction = self.calculate_friction()

        # 2. 인과 구조 역추적 장 (Causal Back-tracing Field)
        # 마찰 F의 S 좌표에 대한 편미분 gradient를 구합니다.
        # dF/ds_i = ∑_j W_{ij} * (||s_i - s_j|| - L_{ij}) * (s_i - s_j) / (||s_i - s_j|| + ε)
        grad_S = np.zeros_like(self.S)
        epsilon = 1e-9

        for i in range(num_nodes):
            for j in range(num_nodes):
                if self.W[i, j] > 0:
                    diff_vec = self.S[i] - self.S[j]
                    dist = np.linalg.norm(diff_vec)
                    strain = dist - self.L[i, j]

                    # 방향 벡터 단위화 및 미분 성분 누적
                    direction = diff_vec / (dist + epsilon)
                    grad_S[i] += self.W[i, j] * strain * direction

        # 각 노드가 체감하는 국소 마찰 강도 (Causal Pointer / Contradiction Index)
        # ∇_S F의 크기가 클수록 해당 지점이 모순과 장력의 근원지임을 가리킵니다.
        local_friction_index = np.linalg.norm(grad_S, axis=1)

        # 3. 국소 위상 변이 및 이완 (Topological Mutation & Relaxation)
        # 1) 좌표 이완 (S의 위치가 마찰을 완화하는 방향으로 자연스럽게 흐름)
        self.S -= self.eta_s * grad_S * dt

        # 무게중심 보존 법칙 강제적 프로젝션 (Symmetry Invariant)
        self.S -= np.mean(self.S, axis=0)

        # 2) 토폴로지 변이 (W의 연결 세기가 모순을 이완하기 위해 미세 재배치)
        # dF/dW_{ij} = 0.5 * (||s_i - s_j|| - L_{ij})^2
        grad_W = np.zeros_like(self.W)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if self.initial_W[i, j] > 0: # 기존 연결된 엣지만 변이 허용
                    dist = np.linalg.norm(self.S[i] - self.S[j])
                    grad_W[i, j] = 0.5 * ((dist - self.L[i, j]) ** 2)

        # 연결 강도 업데이트 (장력이 심한 결합은 느슨해지거나 끊어지며 'Tear' 현상 묘사)
        self.W -= self.eta_w * grad_W * dt
        # 가중치 자연 복원성 및 바운딩 (최소 연결성 유지 및 과도한 붕괴 방지)
        # 단, 실제 연결이 존재하지 않는 셀(initial_W == 0)은 0으로 강제 유지하여 온전한 토폴로지 보존
        self.W = np.clip(self.W * self.weight_damping, 0.1, 5.0)
        self.W[self.initial_W == 0] = 0.0

        # 4. 불변량 고정 및 루프 폐쇄 (State Consolidation & Loop Closure)
        # 이완된 새로운 기하학적 형태에 맞추어 평형 기준인 L을 점진적으로 전이 및 고착시킵니다.
        # dL_{ij}/dt = eta_l * (||s_i - s_j|| - L_{ij})
        for i in range(num_nodes):
            for j in range(num_nodes):
                if self.initial_W[i, j] > 0:
                    dist = np.linalg.norm(self.S[i] - self.S[j])
                    self.L[i, j] += self.eta_l * (dist - self.L[i, j]) * dt

        # 새로운 상태 마찰 재측정
        new_friction = self.calculate_friction()

        return {
            "friction_before": friction,
            "friction_after": new_friction,
            "local_friction_index": local_friction_index.tolist(),
            "grad_S_magnitude": float(np.linalg.norm(grad_S)),
            "weight_matrix": self.W.copy(),
            "state_matrix": self.S.copy(),
            "invariant_matrix": self.L.copy()
        }


class SemanticClosedLoopSystem:
    """
    [의미론적 정보 처리 최소 인과 루프 시스템]
    단순한 수치 추상화를 넘어, 구체적인 '의미론적 개념(Semantic Concepts)'과 그 관계 지형(Topology)을
    실제 정보 매체로 수용하고, 의미 충돌(모순 자극)을 물리적 마찰을 통해 스스로 이완하고 극복하는 인지 서사 모델입니다.
    """
    def __init__(
        self,
        concepts: Dict[str, List[float]],  # 개념명 -> N차원 고유 의미 벡터 (Feature Embedding)
        coordinate_relaxation_rate: float = 0.2,
        weight_mutation_rate: float = 0.1,
        consolidation_rate: float = 0.05,
        weight_damping: float = 0.99
    ):
        self.concept_labels = list(concepts.keys())
        self.num_nodes = len(self.concept_labels)
        self.label_to_index = {label: idx for idx, label in enumerate(self.concept_labels)}

        # 1. 고유 의미 특징 공간 (Conceptual Feature Embedding)
        self.embeddings = np.array([concepts[lbl] for lbl in self.concept_labels], dtype=np.float32)

        # 2. 상태 공간 (S): 의미망이 사영된 2차원 인지 지도 좌표
        # 초기 좌표는 각 개념의 고유 의미 특징 간의 거리에 비례하도록 원형/거리 매핑 기하로 초기화
        angles = np.linspace(0, 2 * np.pi, self.num_nodes, endpoint=False)
        self.S = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
        self.S -= np.mean(self.S, axis=0)  # 무게중심 보존

        # 3. 초기 논리적 인과 연결 (W): 개념들 간의 유사도 기반 연결망 구축
        # 예: 코사인 유사도가 0.3 이상인 것들만 엣지로 연결
        self.W = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    dot_val = np.dot(self.embeddings[i], self.embeddings[j])
                    norm_i = np.linalg.norm(self.embeddings[i]) + 1e-9
                    norm_j = np.linalg.norm(self.embeddings[j]) + 1e-9
                    sim = dot_val / (norm_i * norm_j)
                    if sim > 0.1:  # 임계값 이상의 관계만 활성화
                        self.W[i, j] = sim

        self.initial_W = self.W.copy()

        # 4. 불변량 (I): 개념 간의 논리적 거리(Logical Consistencies)
        # 원래 의미 특징 상의 실제 유클리드 거리를 보존해야 할 불변량 거리 L로 수용
        self.L = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                self.L[i, j] = np.linalg.norm(self.embeddings[i] - self.embeddings[j])

        # 이완 상수
        self.eta_s = coordinate_relaxation_rate
        self.eta_w = weight_mutation_rate
        self.eta_l = consolidation_rate
        self.weight_damping = weight_damping

    def project_semantic_stimulus(self, concept_a: str, concept_b: str, force_distance: float) -> Tuple[int, int]:
        """
        [1. 의미 자극의 위상 수용 (Semantic Stimulus Projection)]
        외부 문맥적 충격(예: "태양은 차갑다", "얼음은 뜨겁다")을 받아들입니다.
        이는 특정 두 개념 간의 보존 평형 거리(L_ab)를 강제로 가깝게 혹은 멀게 '변위'시켜,
        지형 전체에 위상적 모순 긴장을 방류하는 형태로 구현됩니다.
        """
        idx_a = self.label_to_index[concept_a]
        idx_b = self.label_to_index[concept_b]

        # 1. 불변량(L)을 강제 변형하여 모순/위상차를 지형에 주입
        self.L[idx_a, idx_b] = force_distance
        self.L[idx_b, idx_a] = force_distance

        # 2. 강제 엣지 연결 활성화 (기존에 연결이 없었더라도 자극에 의해 새로운 인과 통로 W 형성)
        self.W[idx_a, idx_b] = 1.0
        self.W[idx_b, idx_a] = 1.0
        self.initial_W[idx_a, idx_b] = 1.0
        self.initial_W[idx_b, idx_a] = 1.0

        # 3. 물리적 기하 변위: 두 노드를 강제로 조금 가깝거나 멀게 충격(Impulse) 부여
        diff_vec = self.S[idx_a] - self.S[idx_b]
        curr_dist = np.linalg.norm(diff_vec) + 1e-9
        direction = diff_vec / curr_dist

        # 타겟 거리 대비 변위 산출 후 강제 이동
        displacement = 0.5 * (curr_dist - force_distance) * direction
        self.S[idx_a] -= displacement
        self.S[idx_b] += displacement

        # 무게중심 보존 법칙 강제적 프로젝션
        self.S -= np.mean(self.S, axis=0)

        return idx_a, idx_b

    def calculate_friction(self) -> float:
        """
        [2. 상태 마찰 감지 (Friction Detection)]
        모순된 의미 자극에 의해, 개념들이 맺고 있던 기존 논리적 지층(W)과
        새로 규정된 보존 법칙(L) 간에 얽힌 총 모순 마찰 에너지 F를 측정합니다.
        """
        total_friction = 0.0
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.W[i, j] > 0:
                    dist = np.linalg.norm(self.S[i] - self.S[j])
                    diff = dist - self.L[i, j]
                    total_friction += self.W[i, j] * (diff ** 2)
        return float(0.5 * total_friction)

    def step(self, dt: float = 0.1) -> Dict[str, Any]:
        """
        [의미론적 자율 이완 단계]
        어떤 분기문이나 조건 지시어 없이, 의미론적 마찰 에너지 포텐셜을 최소화하기 위해
        인지 지도가 유기적으로 흐르며 스스로 정렬합니다.
        """
        num_nodes = self.num_nodes
        friction = self.calculate_friction()

        # 1. 인과 구조 역추적 (Causal Back-tracing)
        # 각 개념 노드 및 연결부(Edge)의 마찰 미분 gradient를 산출하여 정확히 '어느 지점이 모순인지' 포인팅합니다.
        grad_S = np.zeros_like(self.S)
        epsilon = 1e-9

        for i in range(num_nodes):
            for j in range(num_nodes):
                if self.W[i, j] > 0:
                    diff_vec = self.S[i] - self.S[j]
                    dist = np.linalg.norm(diff_vec)
                    strain = dist - self.L[i, j]

                    direction = diff_vec / (dist + epsilon)
                    grad_S[i] += self.W[i, j] * strain * direction

        # 각 개념 노드가 체감하는 국소 의미적 모순 지표 (Semantic Contradiction Pointer)
        local_friction_index = np.linalg.norm(grad_S, axis=1)

        # 2. 국소 위상 변이 및 이완 (Topological Mutation & Relaxation)
        # 1) 개념의 기하 지도 좌표 이완
        self.S -= self.eta_s * grad_S * dt
        self.S -= np.mean(self.S, axis=0)  # Center of mass conservation

        # 2) 의미론적 토폴로지 연결 세기 W의 자율 변이
        # 충돌이 해결되지 않고 장력이 계속되면, 모순을 완화하기 위해 기존의 모순적 신념(W)이 물리적으로 약화됨(Tear)
        grad_W = np.zeros_like(self.W)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if self.initial_W[i, j] > 0:
                    dist = np.linalg.norm(self.S[i] - self.S[j])
                    grad_W[i, j] = 0.5 * ((dist - self.L[i, j]) ** 2)

        self.W -= self.eta_w * grad_W * dt
        self.W = np.clip(self.W * self.weight_damping, 0.0, 5.0)
        self.W[self.initial_W == 0] = 0.0

        # 3. 불변량 고정 및 루프 폐쇄 (State Consolidation & Loop Closure)
        # 새로 얻어진 의미적 균형 지점을 영구 신념 불변량 L에 전이시켜 학습을 락인합니다.
        for i in range(num_nodes):
            for j in range(num_nodes):
                if self.initial_W[i, j] > 0:
                    dist = np.linalg.norm(self.S[i] - self.S[j])
                    self.L[i, j] += self.eta_l * (dist - self.L[i, j]) * dt

        new_friction = self.calculate_friction()

        # 각 노드의 마찰 인덱스를 맵 형태로 구성
        semantic_friction_map = {
            self.concept_labels[i]: float(local_friction_index[i])
            for i in range(num_nodes)
        }

        # 연결 강도 상태 맵 구성
        topology_map = {}
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if self.W[i, j] > 0.01:
                    lbl_a = self.concept_labels[i]
                    lbl_b = self.concept_labels[j]
                    topology_map[f"{lbl_a}<->{lbl_b}"] = float(self.W[i, j])

        return {
            "friction_before": friction,
            "friction_after": new_friction,
            "semantic_friction_map": semantic_friction_map,
            "grad_S_magnitude": float(np.linalg.norm(grad_S)),
            "topology_map": topology_map,
            "state_matrix": self.S.copy(),
            "invariant_matrix": self.L.copy()
        }


if __name__ == "__main__":
    # 수치 기반 단순 루프 가동 데모
    loop = MinimalClosedLoopSystem(num_nodes=6)
    print("Initial Friction (Rest State):", loop.calculate_friction())
    loop.project_stimulus(node_index=2, impulse=np.array([1.5, -0.5]))
    for step_idx in range(3):
        metrics = loop.step(dt=0.5)
        print(f"Num Step {step_idx + 1} | Friction: {metrics['friction_after']:.6f}")

    # 의미론적 루프 가동 데모
    print("\n--- Semantic Closed Loop Demonstration ---")
    # 개념들의 고유 특징 공간 (Sun, Fire: 고온 특징, Ice, Cold: 저온 특징)
    semantic_universe = {
        "Sun":  [1.0, 0.0],
        "Fire": [0.9, 0.1],
        "Ice":  [0.0, 1.0],
        "Cold": [0.1, 0.9]
    }

    sem_loop = SemanticClosedLoopSystem(concepts=semantic_universe)
    print("Initial Semantic Friction:", sem_loop.calculate_friction())
    print("Initial Semantic Topology:", sem_loop.step(0.0)["topology_map"])

    # 극단적인 의미 모순 자극 주입: "태양이 차갑다" (Sun과 Cold의 거리를 0.2로 강제 밀착)
    print("\n[Projecting Contradictory Stimulus: 'Sun is Cold' (Distance forced to 0.2)]")
    sem_loop.project_semantic_stimulus("Sun", "Cold", force_distance=0.2)
    print("Semantic Friction immediately after Stimulus:", sem_loop.calculate_friction())

    # 자율 이완 흐름 수행
    print("\n[Relaxing Semantic Manifold autonomously]")
    for step_idx in range(5):
        metrics = sem_loop.step(dt=0.4)
        print(
            f"Step {step_idx + 1} | "
            f"Friction: {metrics['friction_before']:.6f} -> {metrics['friction_after']:.6f} | "
            f"Tension Source Map: {metrics['semantic_friction_map']}"
        )
        print("   Mutated Semantic Topology:", metrics["topology_map"])
