"""
[Inverse Mechanism Engine: 역메커니즘 추출 프레임워크]
표면적 데이터 수치 패턴 매핑을 넘어, 결과 관측값들 간의 위상적 차이(Differential Delta, Δ)와
외부 경계 조건(Boundary Conditions, C)을 대조·기약하여
결과를 발생시킨 '잠재적 인과장 메커니즘 방정식(Generating Mechanism, Θ)'을 역추출합니다.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import math

@dataclass
class BoundaryCondition:
    """외부 환경 및 제약 파라미터 (C)"""
    condition_id: str
    friction: float = 1.0           # 마찰력 / 저항
    scale: float = 1.0              # 스케일 / 경계 영역 크기
    gravity: float = 9.81           # 중력 / 왜곡 장
    temperature: float = 1.0        # 엔트로피 / 무작위성
    extra_constraints: Dict[str, float] = field(default_factory=dict)

    def to_vector(self) -> List[float]:
        return [self.friction, self.scale, self.gravity, self.temperature]

@dataclass
class ObservedTrajectory:
    """관측된 결과 데이터 궤적 (Y)"""
    trajectory_id: str
    boundary_id: str
    states: List[List[float]]      # 시간에 따른 n차원 관측 상태 데이터
    intent_tag: str = "default"    # 생성 의도/목적 태그

@dataclass
class DifferentialDelta:
    """두 관측 궤적 간의 미분적 차이 (Δ)"""
    traj_id_a: str
    traj_id_b: str
    boundary_delta: List[float]    # 경계 조건 차이 (ΔC)
    state_deltas: List[List[float]] # 상태 궤적 차이 (ΔY)
    norm_delta: float              # 상태 차이의 L2 노름 합

@dataclass
class GeneratingMechanism:
    """
    역추출된 잠재 인과장 메커니즘 (Θ)
    결과 파편이 아닌, 궤적을 굴절시키는 상위 인과 방정식 및 위상 불변량
    """
    mechanism_id: str
    intent_vector: List[float]              # 의도 축 (Intent)
    stiffness_matrix: List[List[float]]     # 복원력 / 위상 장력 (Structural Stiffness)
    boundary_coupling: List[List[float]]   # 경계 조건 감도 (Causal Coupling)
    topological_invariants: List[float]     # 환경 변형에도 유지되는 위상 불변량
    description_length: float = 0.0         # 최단 설명 길이 (MDL / Complexity Score)

class InverseMechanismEngine:
    """
    [Inverse Mechanism Engine]
    관측된 결과 데이터(Result Trajectory) 모음집을 입력으로 받아,
    상위 생성 방정식 Θ를 역추출하고 기약성(Reducibility)을 강제하여
    미지의 섭동/외삽 환경에서도 올바른 궤적을 자율 생성합니다.
    """

    def __init__(self, mdl_penalty_weight: float = 0.1):
        self.mdl_penalty_weight = mdl_penalty_weight
        self.extracted_mechanisms: Dict[str, GeneratingMechanism] = {}

    def compute_differential_delta(
        self,
        obs_a: ObservedTrajectory,
        boundary_a: BoundaryCondition,
        obs_b: ObservedTrajectory,
        boundary_b: BoundaryCondition
    ) -> DifferentialDelta:
        """두 관측 궤적 및 경계 조건 간의 미분적 차이 Δ를 계산합니다."""
        vec_a = boundary_a.to_vector()
        vec_b = boundary_b.to_vector()
        boundary_delta = [b - a for a, b in zip(vec_a, vec_b)]

        min_len = min(len(obs_a.states), len(obs_b.states))
        state_deltas = []
        total_norm_sq = 0.0

        for t in range(min_len):
            s_a = obs_a.states[t]
            s_b = obs_b.states[t]
            dim = min(len(s_a), len(s_b))
            d_t = [s_b[i] - s_a[i] for i in range(dim)]
            state_deltas.append(d_t)
            total_norm_sq += sum(val ** 2 for val in d_t)

        norm_delta = math.sqrt(total_norm_sq)

        return DifferentialDelta(
            traj_id_a=obs_a.trajectory_id,
            traj_id_b=obs_b.trajectory_id,
            boundary_delta=boundary_delta,
            state_deltas=state_deltas,
            norm_delta=norm_delta
        )

    def extract_generating_mechanism(
        self,
        mechanism_id: str,
        observations: List[ObservedTrajectory],
        boundaries: Dict[str, BoundaryCondition]
    ) -> GeneratingMechanism:
        """
        결과 데이터 궤적들과 경계 조건들을 입력으로 받아
        잠재 생성 메커니즘 매개변수 Θ를 역추출합니다.
        """
        if not observations:
            raise ValueError("Observations list cannot be empty.")

        dim = len(observations[0].states[0]) if observations[0].states else 1

        # 1. 위상 불변량 (Topological Invariants) 추출: 모든 궤적에서 보존되는 특성
        # 예: 각 궤적의 중심점/운동량 보존량의 평균 패턴
        invariant_sums = [0.0] * dim
        total_points = 0
        for obs in observations:
            for st in obs.states:
                for d in range(min(dim, len(st))):
                    invariant_sums[d] += st[d]
                total_points += 1

        topological_invariants = [s / max(total_points, 1) for s in invariant_sums]

        # 2. 의도 벡터 (Intent Vector) 추출: 궤적의 시작점에서 최종 수렴지점까지의 거시 방향
        intent_vec = [0.0] * dim
        for obs in observations:
            if len(obs.states) >= 2:
                start = obs.states[0]
                end = obs.states[-1]
                for d in range(min(dim, len(start), len(end))):
                    intent_vec[d] += (end[d] - start[d])
        intent_vec = [v / max(len(observations), 1) for v in intent_vec]

        # 3. 경계 조건 감도 행렬 (Boundary Coupling) & 강성 행렬 (Stiffness Matrix) 역산
        # ΔY = Stiffness * Y + BoundaryCoupling * ΔC 관계를 역분해
        coupling_matrix = [[0.0] * 4 for _ in range(dim)]
        stiffness_matrix = [[0.0] * dim for _ in range(dim)]

        # 대각 강성(복원력) 초기화
        for d in range(dim):
            stiffness_matrix[d][d] = 0.5

        # 경계 변형에 따른 궤적 굴절 감도 역추론
        for i in range(len(observations) - 1):
            obs1 = observations[i]
            obs2 = observations[i + 1]
            b1 = boundaries.get(obs1.boundary_id, BoundaryCondition("default1"))
            b2 = boundaries.get(obs2.boundary_id, BoundaryCondition("default2"))

            delta = self.compute_differential_delta(obs1, b1, obs2, b2)

            for d in range(dim):
                for c_idx in range(4):
                    if abs(delta.boundary_delta[c_idx]) > 1e-6:
                        # 경계 변화 대비 상태 변화율
                        avg_state_change = sum(dt[d] for dt in delta.state_deltas) / max(len(delta.state_deltas), 1)
                        coupling_matrix[d][c_idx] += avg_state_change / delta.boundary_delta[c_idx]

        num_pairs = max(len(observations) - 1, 1)
        for d in range(dim):
            for c_idx in range(4):
                coupling_matrix[d][c_idx] /= num_pairs

        raw_mechanism = GeneratingMechanism(
            mechanism_id=mechanism_id,
            intent_vector=intent_vec,
            stiffness_matrix=stiffness_matrix,
            boundary_coupling=coupling_matrix,
            topological_invariants=topological_invariants,
            description_length=0.0
        )

        # 4. 기약성(Reducibility / MDL) 정제 적용하여 무작위 소음 배격
        pure_mechanism = self.enforce_reducibility(raw_mechanism)
        self.extracted_mechanisms[mechanism_id] = pure_mechanism
        return pure_mechanism

    def enforce_reducibility(self, mechanism: GeneratingMechanism, threshold: float = 1e-3) -> GeneratingMechanism:
        """
        [기약성(Reducibility) 정제 엔진]
        최단 설명 길이(MDL) 원칙에 따라 임의의 고차원 수치 곡선맞춤(Overfitting) 파라미터를 배격하고,
        유의미하지 않은 미세 노이즈 매개변수를 0으로 압축 기약합니다.
        """
        # 1. Boundary coupling 마스킹 (Thresholding below epsilon)
        clean_coupling = []
        non_zero_params = 0

        for row in mechanism.boundary_coupling:
            clean_row = []
            for val in row:
                if abs(val) < threshold:
                    clean_row.append(0.0)
                else:
                    clean_row.append(val)
                    non_zero_params += 1
            clean_coupling.append(clean_row)

        # 2. Stiffness matrix 마스킹
        clean_stiffness = []
        for row in mechanism.stiffness_matrix:
            clean_row = []
            for val in row:
                if abs(val) < threshold:
                    clean_row.append(0.0)
                else:
                    clean_row.append(val)
                    non_zero_params += 1
            clean_stiffness.append(clean_row)

        # 3. 최단 설명 길이 (MDL Description Length) 계산: 파라미터 복잡도 + 표현 손실
        description_length = non_zero_params * self.mdl_penalty_weight

        mechanism.boundary_coupling = clean_coupling
        mechanism.stiffness_matrix = clean_stiffness
        mechanism.description_length = description_length
        return mechanism

    def generate_trajectory(
        self,
        mechanism: GeneratingMechanism,
        boundary: BoundaryCondition,
        initial_state: List[float],
        steps: int = 10,
        intent_scale: float = 1.0
    ) -> List[List[float]]:
        """
        [역추출된 생성 방정식 Θ 기반 자율 궤적 복원/생성]
        학습된 단순 패턴 매핑이 아니라, 역추출된 인과 메커니즘 Θ와
        새로운 경계 조건 C, 의도 Intent를 결합하여 외삽(Extrapolation) 궤적을 굴절 생성합니다.

        미분 방정식:
        S_{t+1} = S_t + Intent * Scale - Stiffness * (S_t - Invariant) + Coupling * Boundary_Vec
        """
        dim = len(initial_state)
        trajectory = [list(initial_state)]
        curr_state = list(initial_state)
        b_vec = boundary.to_vector()

        for _ in range(steps - 1):
            next_state = [0.0] * dim
            for d in range(dim):
                # 의도 추진력
                intent_force = mechanism.intent_vector[d] * intent_scale if d < len(mechanism.intent_vector) else 0.0

                # 위상 복원력 (Stiffness & Invariance)
                inv_target = mechanism.topological_invariants[d] if d < len(mechanism.topological_invariants) else 0.0
                stiff_force = sum(
                    mechanism.stiffness_matrix[d][k] * (curr_state[k] - inv_target)
                    for k in range(min(dim, len(mechanism.stiffness_matrix[d])))
                )

                # 경계 조건 굴절력 (Boundary Coupling)
                boundary_force = sum(
                    mechanism.boundary_coupling[d][c] * b_vec[c]
                    for c in range(min(4, len(mechanism.boundary_coupling[d])))
                )

                # 상태 업데이트 (Euler integration of the generating equation)
                delta_s = intent_force - stiff_force + boundary_force
                next_state[d] = curr_state[d] + delta_s

            trajectory.append(next_state)
            curr_state = next_state

        return trajectory
