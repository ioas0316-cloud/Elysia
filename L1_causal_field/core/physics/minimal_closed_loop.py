"""
minimal_closed_loop.py
======================
최소 인과 루프 (Minimal Closed Loop) - 무분기 물리-대수적 모델.

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
from typing import Dict, Any, List


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


class GroundedSensoryClosedLoop:
    """
    [양방향 인과 매니폴드 시스템 (Grounded Sensory Closed Loop)]
    원인, 과정, 결과가 동일한 위상 공간 위에서 상호 구속되는 에너지 평형 방정식 Ф(C, P, E) = 0을 만족하는,
    시각(Visual), 열역학(Thermal), 공간(Spatial) 3대 하부 감각 필드와 Coupled Complex Wave-Void Oscillator 기반의
    무분기 자율 물리 이완 폐회로 Substrate.
    """
    def __init__(
        self,
        temperature: float = 1.0,
        cooling_rate: float = 0.95,
        coordinate_relaxation_rate: float = 0.2,
        thermal_adaptation_rate: float = 0.1,
        phase_synchronization_rate: float = 0.15,
        weight_mutation_rate: float = 0.05,
        consolidation_rate: float = 0.01,
        weight_damping: float = 0.99,
        coupling_beta: float = 0.1,
        coupling_gamma: float = 0.1
    ):
        self.node_names = ["Sun", "Cold", "Ice", "Fire"]
        self.num_nodes = len(self.node_names)
        self.name_to_index = {name: idx for idx, name in enumerate(self.node_names)}

        # 1. Spatial Field (S): N x 2 coordinates (originally in a circle)
        angles = np.linspace(0, 2 * np.pi, self.num_nodes, endpoint=False)
        self.S = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
        self.S -= np.mean(self.S, axis=0) # Center of mass conservation

        # 2. Thermal Field (T): Intrinsic heat charge per node
        # Sun (+2.0), Cold (-2.0), Ice (-1.5), Fire (+1.8)
        self.T = np.array([2.0, -2.0, -1.5, 1.8], dtype=np.float32)

        # 3. Visual Field (V): [Flux, Order, Entropy] chromatic signature (Red, Blue, Yellow)
        self.V = np.array([
            [0.9, 0.1, 0.2],  # Sun: Highly flux (Red)
            [0.1, 0.9, 0.2],  # Cold: Highly order (Blue)
            [0.1, 0.8, 0.5],  # Ice: High order, medium entropy (Yellow)
            [0.9, 0.1, 0.6]   # Fire: High flux, medium entropy
        ], dtype=np.float32)

        # 4. Phase Field (theta) & Wave-Void: Coupled rotors
        # Complex state z = A * e^{i * theta}
        self.theta = np.array([0.0, np.pi, np.pi * 0.8, 0.2], dtype=np.float32)
        self.amplitude = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        # 5. Connectivity Matrix (W): Coupled weights
        self.W = np.ones((self.num_nodes, self.num_nodes), dtype=np.float32) * 0.5
        np.fill_diagonal(self.W, 0.0)
        self.initial_W = self.W.copy()

        # 6. Target Invariants (L, H, Psi) representing Rest State constraints
        self.L = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.H = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.Psi = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self._recalculate_target_invariants()

        # Thermodynamic Temperature Phase Transition variables
        self.temperature = temperature
        self.cooling_rate = cooling_rate

        # Adaptation and relaxation rates
        self.eta_s = coordinate_relaxation_rate
        self.eta_t = thermal_adaptation_rate
        self.eta_theta = phase_synchronization_rate
        self.eta_w = weight_mutation_rate
        self.eta_l = consolidation_rate
        self.weight_damping = weight_damping

        # Cross-field coupling coefficients
        self.beta = coupling_beta
        self.gamma = coupling_gamma

        # Hysteresis memory: tracking energy dissipation
        self.dissipated_energy_history = []

    def _recalculate_target_invariants(self) -> None:
        """현재 상태를 바탕으로 위상 불변량 L, H, Psi를 갱신합니다."""
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                self.L[i, j] = np.linalg.norm(self.S[i] - self.S[j])
                self.H[i, j] = self.T[i] - self.T[j]
                self.Psi[i, j] = self.theta[i] - self.theta[j]

    @property
    def presence_field(self) -> np.ndarray:
        """The 'Presence' manifest state wave: A * e^{i * theta}."""
        return self.amplitude * np.exp(1j * self.theta)

    @property
    def void_field(self) -> np.ndarray:
        """The 'Void' complementary vacuum state wave: A * e^{i * (theta + pi)}."""
        return self.amplitude * np.exp(1j * (self.theta + np.pi))

    def project_stimulus(self, target_concept: str, sensory_impulses: dict) -> None:
        """
        [1. 자극의 위상 수용 (Stimulus Projection)]
        외부 자극을 기호나 데이터로 파싱하지 않고, 지정된 타겟 개념의 각 하부 감각 필드에 대한 물리적 변위로 직접 투사합니다.
        """
        if target_concept not in self.name_to_index:
            return
        idx = self.name_to_index[target_concept]

        if "thermal" in sensory_impulses:
            self.T[idx] += float(sensory_impulses["thermal"])

        if "spatial" in sensory_impulses:
            self.S[idx] += np.array(sensory_impulses["spatial"], dtype=np.float32)
            self.S -= np.mean(self.S, axis=0)

        if "phase" in sensory_impulses:
            self.theta[idx] += float(sensory_impulses["phase"])
            self.theta[idx] = (self.theta[idx] + np.pi) % (2 * np.pi) - np.pi

    def calculate_friction(self) -> float:
        """
        [2. 상태 마찰 감지 (Friction Detection)]
        각 필드(공간, 열역학, 위상 파동)의 불일치 장력의 합을 마찰 F로 산출합니다.
        이때 공간적 평형 거리는 열역학 및 위상 불일치에 의해 유동적으로 변하는 유효 평형 거리 tilde_L을 따릅니다.
        """
        total_friction = 0.0

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                w = self.W[i, j]
                if w > 0:
                    dist = np.linalg.norm(self.S[i] - self.S[j])
                    # Effective target length coupled with Thermal and Phase domains
                    tilde_L = self.L[i, j] + self.beta * ((self.T[i] - self.T[j]) ** 2) + self.gamma * (1.0 - np.cos(self.theta[i] - self.theta[j]))

                    diff_s = dist - tilde_L
                    diff_t = (self.T[i] - self.T[j]) - self.H[i, j]
                    diff_p = self.theta[i] - self.theta[j] - self.Psi[i, j]

                    total_friction += w * (diff_s ** 2 + diff_t ** 2 + (1.0 - np.cos(diff_p)))

        return float(0.5 * total_friction)

    def step(self, dt: float = 0.1) -> Dict[str, Any]:
        """
        [자율 물리 이완 단계 (Let it flow!)]
        Langevin thermal noise가 가미된 물리-인지 쌍대 장의 5단계 자율 이완 루프.
        """
        # 1. 마찰 감지
        friction = self.calculate_friction()
        self.dissipated_energy_history.append(friction)

        # 2. 인과 구조 역추적 (Causal Back-tracing via Gradients of coupled potential)
        grad_S = np.zeros_like(self.S)
        grad_T = np.zeros_like(self.T)
        grad_theta = np.zeros_like(self.theta)
        epsilon = 1e-9

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                w_ij = self.W[i, j]
                w_ji = self.W[j, i]

                if w_ij > 0:
                    dist = np.linalg.norm(self.S[i] - self.S[j])
                    tilde_L = self.L[i, j] + self.beta * ((self.T[i] - self.T[j]) ** 2) + self.gamma * (1.0 - np.cos(self.theta[i] - self.theta[j]))
                    strain = dist - tilde_L

                    # Spatial gradient component
                    diff_vec = self.S[i] - self.S[j]
                    direction = diff_vec / (dist + epsilon)
                    grad_S[i] += w_ij * strain * direction

                    # Thermal gradient component (Piezoelectric-like coupling)
                    term_t = (self.T[i] - self.T[j]) - self.H[i, j]
                    # Derivative of F_elastic w.r.t T_i is -w_ij * strain * 2 * beta * (T_i - T_j)
                    grad_T[i] += w_ij * (term_t - 2.0 * self.beta * strain * (self.T[i] - self.T[j]))

                    # Phase gradient component
                    term_p = self.theta[i] - self.theta[j] - self.Psi[i, j]
                    # Derivative of F_elastic w.r.t theta_i is -w_ij * strain * gamma * sin(theta_i - theta_j)
                    grad_theta[i] += w_ij * (0.5 * np.sin(term_p) - self.gamma * strain * np.sin(self.theta[i] - self.theta[j]))

        # 3. 국소 위상 변이 및 이완 (Topological Mutation & Relaxation with Langevin Noise)
        # Langevin thermal fluctuations
        noise_scale = np.sqrt(self.temperature) if self.temperature > 0 else 0.0

        # Spatial relaxation & noise
        s_noise = np.random.normal(0, 0.05 * noise_scale, size=self.S.shape) if noise_scale > 0 else 0.0
        self.S -= (self.eta_s * grad_S * dt - s_noise)
        self.S -= np.mean(self.S, axis=0) # 무게중심 보존

        # Thermal relaxation & noise
        t_noise = np.random.normal(0, 0.05 * noise_scale, size=self.T.shape) if noise_scale > 0 else 0.0
        self.T -= (self.eta_t * grad_T * dt - t_noise)

        # Phase relaxation & noise
        theta_noise = np.random.normal(0, 0.05 * noise_scale, size=self.theta.shape) if noise_scale > 0 else 0.0
        self.theta -= (self.eta_theta * grad_theta * dt - theta_noise)
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        # 4. 연결 강도/위상 변이 (Weight Mutation)
        grad_W = np.zeros_like(self.W)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.initial_W[i, j] > 0:
                    dist = np.linalg.norm(self.S[i] - self.S[j])
                    tilde_L = self.L[i, j] + self.beta * ((self.T[i] - self.T[j]) ** 2) + self.gamma * (1.0 - np.cos(self.theta[i] - self.theta[j]))
                    diff_s = dist - tilde_L
                    diff_t = (self.T[i] - self.T[j]) - self.H[i, j]
                    diff_p = self.theta[i] - self.theta[j] - self.Psi[i, j]
                    grad_W[i, j] = 0.5 * ((diff_s ** 2) + (diff_t ** 2) + (1.0 - np.cos(diff_p)))

        self.W -= self.eta_w * grad_W * dt
        self.W = np.clip(self.W * self.weight_damping, 0.05, 5.0)
        self.W[self.initial_W == 0] = 0.0

        # 5. 불변량 고정 및 루프 폐쇄 (Consolidation of L, H, Psi)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.initial_W[i, j] > 0:
                    # Spatial consolidation
                    dist = np.linalg.norm(self.S[i] - self.S[j])
                    self.L[i, j] += self.eta_l * (dist - self.L[i, j]) * dt

                    # Thermal consolidation
                    self.H[i, j] += self.eta_l * ((self.T[i] - self.T[j]) - self.H[i, j]) * dt

                    # Phase consolidation
                    diff_p = self.theta[i] - self.theta[j]
                    phase_diff_error = (diff_p - self.Psi[i, j] + np.pi) % (2 * np.pi) - np.pi
                    self.Psi[i, j] += self.eta_l * phase_diff_error * dt
                    self.Psi[i, j] = (self.Psi[i, j] + np.pi) % (2 * np.pi) - np.pi

        # Cooling down (Thermodynamic Phase Transition)
        self.temperature *= (1.0 - (1.0 - self.cooling_rate) * dt)

        # 새로운 상태 마찰 재측정
        new_friction = self.calculate_friction()

        return {
            "friction_before": friction,
            "friction_after": new_friction,
            "local_friction_index": (np.linalg.norm(grad_S, axis=1) + np.abs(grad_T) + np.abs(grad_theta)).tolist(),
            "grad_S_magnitude": float(np.linalg.norm(grad_S)),
            "weight_matrix": self.W.copy(),
            "state_matrix": self.S.copy(),
            "invariant_matrix": self.L.copy(),
            "temperature": self.temperature
        }


if __name__ == "__main__":
    # 간단한 작동 및 수렴 흐름 확인 데모
    loop = MinimalClosedLoopSystem(num_nodes=6)
    print("Initial Friction (Rest State):", loop.calculate_friction())

    # 1. 자극의 위상 수용 (Stimulus Projection): 2번 노드를 바깥쪽으로 강하게 Perturb
    print("\n[Stimulus Projected onto Node 2]")
    loop.project_stimulus(node_index=2, impulse=np.array([1.5, -0.5]))

    print("Friction immediately after Perturbation:", loop.calculate_friction())

    # 2. 자율 이완 (Let it flow!)
    print("\n[Autonomous Relaxation Steps]")
    for step_idx in range(5):
        metrics = loop.step(dt=0.5)
        print(
            f"Step {step_idx + 1} | "
            f"Friction: {metrics['friction_before']:.6f} -> {metrics['friction_after']:.6f} | "
            f"Total Grad: {metrics['grad_S_magnitude']:.6f}"
        )
        print("   Local Friction Index:", [round(x, 4) for x in metrics['local_friction_index']])
