"""
ga_rotor_field.py
=================
기하 대수 로터 필드(GA Rotor Fields) 및 틱리스 연속 이벤트 기반 시뮬레이션 엔진.

핵심 설계 사상 (From THE_ABSOLUTE_COMMANDMENT.md & ROADMAP.md):
- "Do not calculate, let it flow." (계산하지 말고 흐르게 하라)
- 틱(Tick) 단위의 불연속적 갱신을 지양하고, 상태를 시간 t에 대한 연속 해석 함수로 정의하여 연산량 O(0)~O(1) 실현.
- 이산적 충돌(Discrete Collision) 패러다임을 극복하고, 기하 대수(Geometric Algebra)의 로터(Rotor)와
  리 대수(Lie Algebra) 상의 바이벡터(Bivector) 가중 선형 중첩을 사용해 다자간 구속조건을 완벽히 충족하는 연속체 유체 흐름을 형성.
- SoA(Struct of Arrays) 메모리 레이아웃 및 핑퐁 더블 버퍼링(Ping-Pong Double Buffering) 모사를 통해 초병렬 하드웨어 호환성 확보.
- 사고 잠재 공간(Cognitive Latent Space) 상에서 논리적 모순/환각 영역(SDF Obstacle)을 유연하게 우회하는 Thought Trajectory 지원.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class GARotorFieldSystem:
    """
    [기하 대수 로터 필드 & 틱리스 시뮬레이션 시스템]
    SoA(Struct of Arrays) 구조와 핑퐁 버퍼, pre-baked SDF, GA Rotor / Lie Algebra Fusion을 장착한
    우주적 차원의 연속성 시뮬레이션 모태기반(Persistent Substrate).
    """
    def __init__(
        self,
        num_agents: int = 100,
        dims: int = 2,
        influence_radius: float = 1.2,
        barrier_p: float = 2.0,
        sigma: float = 0.5
    ):
        self.num_agents = num_agents
        self.dims = dims
        self.influence_radius = influence_radius
        self.barrier_p = barrier_p
        self.sigma = sigma

        # 1. SoA (Struct of Arrays) 메모리 구조
        # 핑퐁 더블 버퍼링 (Buffer 0, Buffer 1)
        self.current_buffer_idx = 0
        self.pos_buffers = [
            np.zeros((num_agents, dims), dtype=np.float32),
            np.zeros((num_agents, dims), dtype=np.float32)
        ]

        self.preferred_velocities = np.zeros((num_agents, dims), dtype=np.float32)
        self.output_velocities = np.zeros((num_agents, dims), dtype=np.float32)
        self.radii = np.ones(num_agents, dtype=np.float32) * 0.2

        # 로터 방향 버퍼: 2D에서는 복소수(float2), 3D에서는 쿼터니언(float4)
        rotor_size = 2 if dims == 2 else (4 if dims == 3 else dims)
        self.orientations = np.zeros((num_agents, rotor_size), dtype=np.float32)
        if dims == 2:
            self.orientations[:, 0] = 1.0  # Real part = 1.0 (cos 0)
        elif dims == 3:
            self.orientations[:, 3] = 1.0  # w = 1.0 (identity quaternion)

        # 2. 틱리스 연속 궤적 매개변수 (t에 대한 연속 함수를 위한 파라미터 백업)
        # x(t) = x_0 + v * (t - t_0)
        self.trajectory_x0 = np.zeros((num_agents, dims), dtype=np.float32)
        self.trajectory_v = np.zeros((num_agents, dims), dtype=np.float32)
        self.trajectory_t0 = np.zeros(num_agents, dtype=np.float32)

        # 3. 사전 정보화: Pre-baked SDF (Signed Distance Field) / Lookup Table (LUT)
        # 평면 격자 형태의 pre-baked SDF 데이터 필드
        self.obstacles: List[Dict[str, Any]] = []
        self.sdf_lut: Optional[np.ndarray] = None
        self.sdf_bounds: Optional[Tuple[float, float, float, float]] = None # x_min, x_max, y_min, y_max
        self.sdf_res: Optional[Tuple[int, int]] = None

        # 4. 대상(Target) 목적지 목록 (Parametric Injection을 위한 포인터/키 테이블)
        self.targets: Dict[str, np.ndarray] = {}
        self.agent_target_keys = [""] * num_agents

    @property
    def positions(self) -> np.ndarray:
        """현재 활성화된 핑퐁 위치 버퍼를 반환합니다."""
        return self.pos_buffers[self.current_buffer_idx]

    @positions.setter
    def positions(self, val: np.ndarray):
        self.pos_buffers[self.current_buffer_idx] = val.astype(np.float32)

    @property
    def next_positions(self) -> np.ndarray:
        """다음 핑퐁 위치 버퍼를 반환합니다."""
        return self.pos_buffers[1 - self.current_buffer_idx]

    @next_positions.setter
    def next_positions(self, val: np.ndarray):
        """다음 핑퐁 위치 버퍼를 설정합니다."""
        self.pos_buffers[1 - self.current_buffer_idx] = val.astype(np.float32)

    def swap_buffers(self):
        """핑퐁 버퍼의 순서를 교체하여 Race Condition 없이 Frame N -> N+1 전환을 완료합니다."""
        self.current_buffer_idx = 1 - self.current_buffer_idx

    # -------------------------------------------------------------
    # SDF (Signed Distance Field) Pre-baking & Parametric Injection
    # -------------------------------------------------------------
    def add_obstacle(self, center: np.ndarray, radius: float):
        """환경에 원형/구형 장애물(모순 영역)을 추가합니다."""
        self.obstacles.append({
            "center": np.array(center, dtype=np.float32),
            "radius": radius
        })

    def pre_bake_sdf(self, x_range: Tuple[float, float], y_range: Tuple[float, float], resolution: Tuple[int, int]):
        """
        [공간의 정보화: Pre-baking / Offline SDF Structuring]
        장애물 정보를 바탕으로 Signed Distance Field를 격자(LUT) 형태로 미리 구워 둡니다.
        실시간 루프에서는 O(1) 수준의 텍스처 조회(Grid Lookup)로 최단 거리와 그래디언트를 획득합니다.
        """
        self.sdf_bounds = (x_range[0], x_range[1], y_range[0], y_range[1])
        self.sdf_res = resolution
        w, h = resolution

        self.sdf_lut = np.zeros((h, w), dtype=np.float32)

        xs = np.linspace(x_range[0], x_range[1], w)
        ys = np.linspace(y_range[0], y_range[1], h)

        for r_idx in range(h):
            for c_idx in range(w):
                pos = np.array([xs[c_idx], ys[r_idx]], dtype=np.float32)
                min_dist = float('inf')

                # 원본 SDF 계산: 장애물 경계면으로부터의 거리 (외부 +, 내부 -)
                for obs in self.obstacles:
                    dist_to_center = np.linalg.norm(pos - obs["center"])
                    dist_to_edge = dist_to_center - obs["radius"]
                    if dist_to_edge < min_dist:
                        min_dist = dist_to_edge

                self.sdf_lut[r_idx, c_idx] = min_dist

    def sample_sdf(self, pos: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        [O(1) SDF Lookup & Bilinear Interpolation]
        Pre-baked SDF LUT에서 이선형 보간(Bilinear Interpolation)을 통해
        임의 좌표에서의 최단 거리와 외향 그래디언트(Gradient)를 복원합니다.
        """
        if self.sdf_lut is None or self.dims != 2:
            # Pre-baked SDF가 없거나 3D인 경우 실시간 해석적(Analytical) SDF로 폴백
            min_dist = float('inf')
            grad = np.zeros_like(pos)
            epsilon = 1e-9

            for obs in self.obstacles:
                diff = pos - obs["center"]
                dist = np.linalg.norm(diff)
                dist_to_edge = dist - obs["radius"]
                if dist_to_edge < min_dist:
                    min_dist = dist_to_edge
                    grad = diff / (dist + epsilon)

            if min_dist == float('inf'):
                return 10.0, np.zeros_like(pos)
            return min_dist, grad

        # 2D LUT Bilinear Interpolation
        x_min, x_max, y_min, y_max = self.sdf_bounds
        w, h = self.sdf_res

        # Grid 좌표로 스케일링
        x_pct = (pos[0] - x_min) / (x_max - x_min)
        y_pct = (pos[1] - y_min) / (y_max - y_min)

        # 바운딩
        x_pct = np.clip(x_pct, 0.0, 0.999)
        y_pct = np.clip(y_pct, 0.0, 0.999)

        col = x_pct * (w - 1)
        row = y_pct * (h - 1)

        c0, r0 = int(np.floor(col)), int(np.floor(row))
        c1, r1 = min(c0 + 1, w - 1), min(r0 + 1, h - 1)

        tx = col - c0
        ty = row - r0

        # 4방향 샘플링
        val_00 = self.sdf_lut[r0, c0]
        val_10 = self.sdf_lut[r0, c1]
        val_01 = self.sdf_lut[r1, c0]
        val_11 = self.sdf_lut[r1, c1]

        # 보간
        dist = (1 - ty) * ((1 - tx) * val_00 + tx * val_10) + ty * ((1 - tx) * val_01 + tx * val_11)

        # 수치 미분을 통한 그래디언트 복원 (Finite Difference)
        # 아주 미세한 변위를 활용해 SDF 거리의 경사를 직접 측정
        step_x = (x_max - x_min) / w
        step_y = (y_max - y_min) / h

        grad_x = (val_10 - val_00) / (step_x + 1e-9)
        grad_y = (val_01 - val_00) / (step_y + 1e-9)
        grad = np.array([grad_x, grad_y], dtype=np.float32)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-6:
            grad /= grad_norm

        return dist, grad

    def register_target(self, key: str, pos: np.ndarray):
        """개념 공간이나 물리 공간 상의 목적지 정답 좌표를 등록합니다."""
        self.targets[key] = np.array(pos, dtype=np.float32)

    def parametric_inject_target(self, agent_idx: int, target_key: str):
        """
        [2. 실시간 변수 대입 (Parametric Injection)]
        유저가 목적지를 교체할 때, 새로이 길찾기 연산을 시작하는 것이 아니라
        단지 Target이라는 Key값으로 정의된 정보 포인터만 교체($O(1)$)합니다.
        """
        if target_key in self.targets:
            self.agent_target_keys[agent_idx] = target_key
            # 선호 속도(v0) 방향 즉시 갱신
            target_pos = self.targets[target_key]
            curr_pos = self.positions[agent_idx]
            v_goal = target_pos - curr_pos
            dist = np.linalg.norm(v_goal)
            if dist > 1e-5:
                self.preferred_velocities[agent_idx] = v_goal / dist
            else:
                self.preferred_velocities[agent_idx] = np.zeros(self.dims, dtype=np.float32)

    # -------------------------------------------------------------
    # GA Rotor & Lie Algebra 가중 합성 핵심 연산
    # -------------------------------------------------------------
    def synthesize_rotor_fields(self):
        """
        [기하 대수 로터 필드 및 리 대수 가중 합성]
        각 에이전트마다 인접 유닛 및 Pre-baked SDF 모순 경계면을 검출하고,
        리 대수(Lie Algebra) 상에서 가중 중첩을 통해 최적의 논리/물리 우회 로터를 복원합니다.
        """
        epsilon = 1e-9
        pos = self.positions
        v0 = self.preferred_velocities

        for i in range(self.num_agents):
            pos_i = pos[i]
            v0_i = v0[i]
            r_i = self.radii[i]

            # 1. 누적기 초기화 (2D의 경우 단일 스칼라 바이벡터, 3D의 경우 float3 vector)
            omega_sum = 0.0 if self.dims == 2 else np.zeros(3, dtype=np.float32)
            weight_sum = 0.0

            # ---------------------------------------------------------
            # (A) Pre-baked SDF 장애물 경계 감지 및 로터 생성
            # ---------------------------------------------------------
            sdf_dist, sdf_grad = self.sample_sdf(pos_i)
            # 영향 영역 체크
            effective_dist = max(sdf_dist - r_i, 1e-5)
            if effective_dist < self.influence_radius:
                # Barrier Weight 계산
                weight = 1.0 / (effective_dist ** self.barrier_p)

                # 외향 그래디언트(sdf_grad)를 모순 회피 벡터로 삼아 바이벡터 생성
                # 2D 외적 (rAB ^ v0)
                if self.dims == 2:
                    # 상대 벡터를 역방향으로 정의하여 장애물에서 밀려나도록 설정
                    r_vec = -sdf_grad * self.influence_radius
                    bivector_area = r_vec[0] * v0_i[1] - r_vec[1] * v0_i[0]
                    bivector_norm = abs(bivector_area) + epsilon
                    b_normalized = bivector_area / bivector_norm

                    # 가까울수록 90도(pi/2) 회전각 형성
                    theta = (np.pi / 2.0) * np.exp(-effective_dist / self.sigma)
                    omega_i = -0.5 * theta * b_normalized

                    omega_sum += weight * omega_i
                    weight_sum += weight
                elif self.dims == 3:
                    # 3D: r_vec ^ v0_i -> bivector is dual to cross product
                    r_vec = -sdf_grad * self.influence_radius
                    b_vector = np.cross(r_vec, v0_i)
                    b_norm = np.linalg.norm(b_vector) + epsilon
                    b_normalized = b_vector / b_norm

                    theta = (np.pi / 2.0) * np.exp(-effective_dist / self.sigma)
                    omega_i = -0.5 * theta * b_normalized

                    omega_sum += weight * omega_i
                    weight_sum += weight

            # ---------------------------------------------------------
            # (B) 인접 유닛 간 구속조건(Non-penetration)에 따른 로터 생성
            # ---------------------------------------------------------
            for j in range(self.num_agents):
                if i == j: continue
                pos_j = pos[j]
                r_j = self.radii[j]

                r_ij = pos_i - pos_j
                dist = np.linalg.norm(r_ij)
                d_min = r_i + r_j

                effective_dist = max(dist - d_min, 1e-5)
                if effective_dist < self.influence_radius:
                    weight = 1.0 / (effective_dist ** self.barrier_p)

                    if self.dims == 2:
                        bivector_area = r_ij[0] * v0_i[1] - r_ij[1] * v0_i[0]
                        # 만약 정면 충돌인 경우 (bivector_area = 0), 회전 축이 대칭이 되어 회전이 상쇄되지 않도록
                        # 인위적인 미세 노이즈나 고유한 외적 대칭성 섭동(Symmetry Perturbation)을 부여합니다.
                        if abs(bivector_area) < 1e-4:
                            bivector_area = 1e-4  # 대칭성 깨뜨리기 (Symmetry breaking)
                        bivector_norm = abs(bivector_area) + epsilon
                        b_normalized = bivector_area / bivector_norm

                        theta = (np.pi / 2.0) * np.exp(-effective_dist / self.sigma)
                        omega_ij = -0.5 * theta * b_normalized

                        omega_sum += weight * omega_ij
                        weight_sum += weight
                    elif self.dims == 3:
                        b_vector = np.cross(r_ij, v0_i)
                        b_norm = np.linalg.norm(b_vector) + epsilon
                        b_normalized = b_vector / b_norm

                        theta = (np.pi / 2.0) * np.exp(-effective_dist / self.sigma)
                        omega_ij = -0.5 * theta * b_normalized

                        omega_sum += weight * omega_ij
                        weight_sum += weight

            # ---------------------------------------------------------
            # (C) 리 대수 융합 및 로터 복원 (Exponential Map)
            # ---------------------------------------------------------
            if weight_sum > 0:
                omega_composite = omega_sum / weight_sum
            else:
                omega_composite = 0.0 if self.dims == 2 else np.zeros(3, dtype=np.float32)

            # 6. Sandwich Product (또는 고속 삼각 회전 연산)을 적용해 최종 속도 도출
            if self.dims == 2:
                # 2 * omega_composite 만큼 회전
                final_angle = 2.0 * omega_composite
                cos_r = np.cos(final_angle)
                sin_r = np.sin(final_angle)

                v_out_x = v0_i[0] * cos_r - v0_i[1] * sin_r
                v_out_y = v0_i[0] * sin_r + v0_i[1] * cos_r
                self.output_velocities[i] = np.array([v_out_x, v_out_y], dtype=np.float32)

                # 로터 상태 저장 (Real, Imaginary)
                self.orientations[i] = np.array([np.cos(omega_composite), np.sin(omega_composite)], dtype=np.float32)
            elif self.dims == 3:
                # 3D: 복원된 리 대수 생성자를 단위 쿼터니언(Rotor)으로 지수 사영
                angle_half = np.linalg.norm(omega_composite)
                if angle_half > 1e-7:
                    axis = omega_composite / angle_half
                    # 쿼터니언 q = (sin(theta/2)*axis, cos(theta/2))
                    # R = e^{-omega_composite} 이므로 sandwich product 회전각은 2 * angle_half
                    q_xyz = np.sin(angle_half) * axis
                    q_w = np.cos(angle_half)
                    self.orientations[i] = np.array([q_xyz[0], q_xyz[1], q_xyz[2], q_w], dtype=np.float32)

                    # 쿼터니언을 사용해 v0_i 회전
                    # q * v0 * q_conj
                    q_v = np.array([v0_i[0], v0_i[1], v0_i[2], 0.0], dtype=np.float32)
                    # 쿼터니언 곱 구현
                    q = self.orientations[i]
                    q_conj = np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)

                    # temp = q * q_v
                    temp_w = -q[0]*q_v[0] - q[1]*q_v[1] - q[2]*q_v[2]
                    temp_x =  q[3]*q_v[0] + q[1]*q_v[2] - q[2]*q_v[1]
                    temp_y =  q[3]*q_v[1] + q[2]*q_v[0] - q[0]*q_v[2]
                    temp_z =  q[3]*q_v[2] + q[0]*q_v[1] - q[1]*q_v[0]

                    # res = temp * q_conj
                    res_x = temp_w*q_conj[0] + temp_x*q_conj[3] + temp_y*q_conj[2] - temp_z*q_conj[1]
                    res_y = temp_w*q_conj[1] + temp_y*q_conj[3] + temp_z*q_conj[0] - temp_x*q_conj[2]
                    res_z = temp_w*q_conj[2] + temp_z*q_conj[3] + temp_x*q_conj[1] - temp_y*q_conj[0]

                    self.output_velocities[i] = np.array([res_x, res_y, res_z], dtype=np.float32)
                else:
                    self.orientations[i] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                    self.output_velocities[i] = v0_i.copy()
            else:
                self.output_velocities[i] = v0_i.copy()

    # -------------------------------------------------------------
    # 틱리스 연속 이벤트 기반 시뮬레이션 (Tickless Simulation)
    # -------------------------------------------------------------
    def initialize_tickless_trajectories(self, t_start: float = 0.0):
        """
        [틱리스 엔진 초기 상태 방정식 주입]
        각 에이전트의 시작 좌표 x_0, 시작 시간 t_0, 그리고 로터 필드를 통해 구해진 최적의 연속 속도 v를 동기화합니다.
        """
        self.synthesize_rotor_fields()
        self.trajectory_x0 = self.positions.copy()
        self.trajectory_v = self.output_velocities.copy()
        self.trajectory_t0.fill(t_start)

    def sample_tickless_positions(self, t: float) -> np.ndarray:
        """
        [O(1) 시간 분석 조회: Continuous Trajectory Sampling]
        엔진 내부에서 틱을 돌리며 적분하지 않고, 단지 지정 시간 t를 파라미터로 주입하여
        모든 유닛의 정확한 좌표를 O(1) 수준으로 즉시 조회해 냅니다.
        x(t) = x_0 + v * (t - t_0)
        """
        dt_array = t - self.trajectory_t0
        # 형상 방송(Broadcasting) 곱셈을 통해 연속적 좌표 추정
        sampled_pos = self.trajectory_x0 + self.trajectory_v * dt_array[:, np.newaxis]
        return sampled_pos

    def trigger_event_update(self, t_current: float, adjusted_agent_indices: Optional[List[int]] = None):
        """
        [이벤트 발생 시점의 상태 방정식 전역/국소 업데이트]
        유저 입력 변화, 병목 구역 진입 등의 '이벤트(Event)' 발생 시에만
        현재 시간 t_current에서의 정확한 좌표를 샘플링한 후, 새로운 x_0, t_0, v를 갱신합니다.
        """
        # 1. 현재 시점의 실시간 위치 계산
        curr_pos = self.sample_tickless_positions(t_current)

        # 2. 핑퐁 더블 버퍼링 활용 (Race condition 차단)
        self.positions = curr_pos

        # 3. 새로운 로터 필드 속도 재합성
        self.synthesize_rotor_fields()

        # 4. 상태 방정식 변수 교체
        if adjusted_agent_indices is None:
            # 전역 업데이트
            self.trajectory_x0 = self.positions.copy()
            self.trajectory_v = self.output_velocities.copy()
            self.trajectory_t0.fill(t_current)
        else:
            # 특정 국소 에이전트들만 효율적으로 부분 업데이트 (Event-driven Optimization)
            for idx in adjusted_agent_indices:
                self.trajectory_x0[idx] = self.positions[idx]
                self.trajectory_v[idx] = self.output_velocities[idx]
                self.trajectory_t0[idx] = t_current

    def step_ping_pong_integration(self, dt: float):
        """
        [전형적인 프레임 단위 이산 시뮬레이션 모사 - 핑퐁 검증용]
        더블 버퍼링을 검증하기 위한 틱 기반 업데이트. Buffer A에서 데이터를 읽어 Buffer B에 갱신한 뒤 스왑합니다.
        """
        self.synthesize_rotor_fields()

        # Ping-pong double buffering write
        self.next_positions = self.positions + self.output_velocities * dt
        self.swap_buffers()


# -------------------------------------------------------------
# Cognitive Latent Space Crossover (사고 잠재 공간 횡단 매니폴드)
# -------------------------------------------------------------
class CognitiveThoughtTrajectory:
    """
    [연속적 잠재 사고 그래프 (Continuous Thought Trajectory)]
    사고의 개념 임베딩 공간에서 논리적 모순이나 환각 구역(SDF Obstacle)을
    기하 대수 로터 필드를 가동해 유연하게 미끄러지듯 회피하는 초고차원 사고 엔진.
    """
    def __init__(self, embedding_dim: int = 128, contradiction_threshold: float = 1.0):
        self.dim = embedding_dim
        self.threshold = contradiction_threshold
        # 모순 장벽들 (Obstacles in latent space)
        self.contradictions: List[Tuple[np.ndarray, float]] = [] # (Center Embedding, Radius)

    def add_contradiction_zone(self, center_emb: np.ndarray, radius: float):
        self.contradictions.append((np.array(center_emb, dtype=np.float32), radius))

    def navigate_thought(self, start_emb: np.ndarray, goal_emb: np.ndarray, steps: int = 50, dt: float = 0.05) -> List[np.ndarray]:
        """
        잠재 공간에서 논리적 파괴를 겪지 않고 회전형 로터를 타며 부드럽게 정답 개념으로 귀착하는 사고 궤적 생성.
        """
        curr = np.array(start_emb, dtype=np.float32)
        goal = np.array(goal_emb, dtype=np.float32)
        trajectory = [curr.copy()]
        epsilon = 1e-9

        for _ in range(steps):
            v_goal = goal - curr
            dist_to_goal = np.linalg.norm(v_goal)
            if dist_to_goal < 0.02:
                break

            v_dir = v_goal / (dist_to_goal + epsilon)
            v_next = v_dir.copy()

            # 다차원 잠재 공간에서의 모순 영향력 누적
            omega_sum = np.zeros_like(curr)
            weight_sum = 0.0

            for center, radius in self.contradictions:
                r_obs = curr - center
                dist_to_obs = np.linalg.norm(r_obs)
                effective_dist = max(dist_to_obs - radius, 1e-5)

                if effective_dist < self.threshold:
                    weight = 1.0 / (effective_dist ** 2)

                    # 고차원 로터: 상대 벡터와 선호 진행 방향의 wedge product로 형성되는 고차원 회전 평면 모사
                    # 2D 평면 사영을 통해 가장 마찰이 높은 주성분 방향의 2차원 회전 수행
                    # 2차원 평면 기저 [u1, u2]를 r_obs와 v_dir로 구축하여 회전 적용
                    u1 = r_obs / (dist_to_obs + epsilon)
                    # u2는 v_dir에서 u1 방향 성분을 제외한 직교 벡터
                    proj = np.dot(v_dir, u1) * u1
                    u2 = v_dir - proj
                    u2_norm = np.linalg.norm(u2)

                    # 정면에서 날아오는 대칭 상황(u2_norm == 0)일 때, 회전을 위한 대칭성 붕괴(Symmetry breaking) 직교 차원 탐색
                    if u2_norm < 1e-6:
                        # 정교하게 직교 차원 하나를 선택하여 방향을 선회
                        u2 = np.zeros_like(v_dir)
                        # 가장 작은 값을 가지는 성분을 찾아 거기에 수직성 부여
                        min_idx = np.argmin(np.abs(u1))
                        u2[min_idx] = 1.0
                        # 직교화 (Gram-Schmidt)
                        u2 = u2 - np.dot(u2, u1) * u1
                        u2_norm = np.linalg.norm(u2)

                    if u2_norm > 1e-6:
                        u2 /= u2_norm
                        # 회전각 도출 (장벽에 가까워질수록 최대 90도 회전하여 완벽한 접선 방향 우회 궤적 완성)
                        theta = (np.pi / 2.0) * np.exp(-effective_dist / 0.5)

                        cos_t = np.cos(theta)
                        sin_t = np.sin(theta)

                        # R = cos(theta) + I*sin(theta) 적용하여 v_dir를 u2 평면 방향으로 부드럽게 꺾음
                        rot_v = cos_t * v_dir + sin_t * u2
                        omega_sum += weight * rot_v
                        weight_sum += weight

            if weight_sum > 0:
                v_next = omega_sum / weight_sum
                v_next_norm = np.linalg.norm(v_next)
                if v_next_norm > 1e-6:
                    v_next /= v_next_norm

            curr += v_next * dt
            trajectory.append(curr.copy())

        return trajectory
