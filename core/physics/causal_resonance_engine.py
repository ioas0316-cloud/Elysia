from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.ndimage import gaussian_filter1d


class EngineStatus(Enum):
    """
    [EngineStatus - 연산 통과 모드]
    RESONANCE_BYPASS: 마찰이 임계값(epsilon) 미만인 공진 상태로, 사고 루프를 우회하여 즉시 행동/기율로 기호 고정.
    INTERNAL_SIMULATION: 마찰 발생 시 내적 시뮬레이터 및 최소 작용 원리로 관측 렌즈 S_t를 재정렬.
    """
    RESONANCE_BYPASS = "RESONANCE_BYPASS"
    INTERNAL_SIMULATION = "INTERNAL_SIMULATION"


@dataclass(frozen=True)
class SymbolicInvariant:
    """
    [SymbolicInvariant - 기호 고정 불변 구조]
    위상 공진 수렴 시 불변 구조 I_t를 고정(Symbolization)하여 반환하는 객체.
    - vector: 수렴된 불변 성분 I_t
    - magnitude: 불변 성분의 크기 ||I_t||
    - resonance_state: 공진 상태 여부 (True: Bypass, False: Internal Simulation 후 반환)
    - friction_energy: 위상 마찰 에너지 E(V_t) = 0.5 * ||V_t||^2
    """
    vector: np.ndarray
    magnitude: float
    resonance_state: bool
    friction_energy: float


class CausalResonanceEngine:
    """
    [CausalResonanceEngine - 최소 인과 엔진]
    세상, 스케일 필터, 위상 마찰 수렴만을 남기고 불필요한 중계 레이어를 모두 제거한 최소 인과 엔진.
    데이터 변환/임베딩을 배제하고, O(1) 국소 위상차 연산 및 마찰 유무에 따른 공진 바이패스/내적 시뮬레이션을 수행한다.
    """

    def __init__(
        self,
        initial_lens: np.ndarray,
        target_boundary: np.ndarray | None = None,
        epsilon: float = 1e-4,
        eta: float = 0.08,
        lmbda: float = 0.02,
        num_probes: int = 12,
        filter_cutoff: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        initial_lens : np.ndarray
            초기 관측 렌즈 벡터 S_0 (1D numpy array, float64).
        target_boundary : np.ndarray | None, optional
            목적 경계 조건 X_future. 지정되지 않을 경우 initial_lens와 동일한 규격의 단위 벡터.
        epsilon : float, default=1e-4
            위상 마찰 수렴 임계값.
        eta : float, default=0.08
            내적 시뮬레이션 렌즈 적응률 (Learning/Adaptation Rate).
        lmbda : float, default=0.02
            최소 작용 렌즈 변형 제동 가중치 (Regularization Weight).
        num_probes : int, default=12
            내적 시뮬레이션 시 탐색할 가상 섭동 경로 개수 (K).
        filter_cutoff : float, default=0.5
            스케일 필터 대역폭 기준 파라미터.
        """
        self.S_t = np.array(initial_lens, dtype=np.float64)

        if target_boundary is not None:
            self.X_future = np.array(target_boundary, dtype=np.float64)
        else:
            self.X_future = np.ones_like(self.S_t, dtype=np.float64)
            norm_val = np.linalg.norm(self.X_future)
            if norm_val > 1e-12:
                self.X_future /= norm_val

        self.epsilon = float(epsilon)
        self.eta = float(eta)
        self.lmbda = float(lmbda)
        self.num_probes = int(num_probes)
        self.filter_cutoff = float(filter_cutoff)

    def scale_filter(self, X_raw: np.ndarray) -> np.ndarray:
        """
        [스케일 필터 모듈 - Scale-Space Sensor Filter]
        목적 경계 X_future에 의거하여 스케일 필터의 관측 대역폭을 동적 지정하고
        유입 신호 X_raw에서 무관한 고주파 잡음 및 미시 변이를 스크리닝한다.
        """
        raw = np.array(X_raw, dtype=np.float64)
        target_norm = np.linalg.norm(self.X_future)
        sigma = self.filter_cutoff / (target_norm + 1e-8)

        # 1D 가우시안 스케일 필터링 (sigma가 0 이하인 경우 원본 유지)
        if sigma <= 0.001:
            return raw

        return gaussian_filter1d(raw, sigma=sigma, mode='nearest')

    def project(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        """
        [직교 정사영]
        신호 X를 관측 렌즈 축 S 상에 정사영하여 공진 성분(I_t)을 추출한다.
        I_t = (<X, S> / (||S||^2 + eps)) * S
        """
        s_norm_sq = np.dot(S, S)
        if s_norm_sq < 1e-12:
            return np.zeros_like(X)
        proj_coeff = np.dot(X, S) / s_norm_sq
        return proj_coeff * S

    def compute_friction(self, X_filtered: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        [위상 공진 게이트 - Phase Resonance Gate]
        1. 불변량 분별: I_t = project(X_filtered, S)
        2. 위상 마찰 포획: V_t = X_filtered - I_t
        3. 마찰 에너지 산출: E(V_t) = 0.5 * ||V_t||^2
        """
        I_t = self.project(X_filtered, S)
        V_t = X_filtered - I_t
        energy = 0.5 * float(np.dot(V_t, V_t))
        return I_t, V_t, energy

    def least_action_simulation(self, X_filtered: np.ndarray, V_t: np.ndarray) -> np.ndarray:
        """
        [최소 작용 내적 시뮬레이터 - Least-Action Internal Simulator]
        마찰 발생 시 K개의 가상 변이 Delta S^{(k)} 후보 중
        작용 적분 S^{(k)} = ||V^{(k)}||^2 + lambda * ||Delta S^{(k)}||^2 를 최소화하는
        최적 변위 eta * Delta S^{(k*)}를 O(1) 탐색한다.
        """
        dim = self.S_t.shape[0]
        v_norm = np.linalg.norm(V_t)

        probes = []

        # 1. 마찰 방향 V_t 의 정방향 및 역방향 가상 변이
        if v_norm > 1e-12:
            unit_v = V_t / v_norm
            probes.append(unit_v)
            probes.append(-unit_v)

        # 2. 국소 가우시안 섭동 탐색 후보 생성
        rng = np.random.default_rng(42)
        while len(probes) < self.num_probes:
            noise = rng.normal(0.0, 1.0, size=dim)
            n_norm = np.linalg.norm(noise)
            if n_norm > 1e-12:
                probes.append(noise / n_norm)

        best_action = float('inf')
        best_delta_S = np.zeros(dim, dtype=np.float64)

        for probe_dir in probes:
            step_delta = self.eta * probe_dir
            # 가상 렌즈 축 S_candidate = S_t + step_delta
            cand_lens = self.S_t + step_delta

            # 가상 마찰 V_cand 계산
            cand_I = self.project(X_filtered, cand_lens)
            cand_V = X_filtered - cand_I

            # 작용 함숫값 S^{(k)} = ||cand_V||^2 + lambda * ||step_delta||^2
            action_val = float(np.dot(cand_V, cand_V)) + self.lmbda * float(np.dot(step_delta, step_delta))

            if action_val < best_action:
                best_action = action_val
                best_delta_S = step_delta

        return best_delta_S

    def step(self, X_raw: np.ndarray) -> tuple[SymbolicInvariant, EngineStatus]:
        """
        [최소 데이터 흐름 파이프라인 스텝]
        1. 스케일 필터링 (스케일 스크리닝)
        2. 직교 분해 (불변량 vs 위상 마찰)
        3. 제로존 바이패스 (E(V_t) < epsilon) vs 사고 루프 전환 (E(V_t) >= epsilon)
        4. 내적 시뮬레이션을 통한 최소 작용 렌즈 재정렬 (S_t += delta_S)
        """
        # 1. 스케일 필터링
        X_filtered = self.scale_filter(X_raw)

        # 2. 직교 분해 및 마찰 에너지 산출
        I_t, V_t, energy = self.compute_friction(X_filtered, self.S_t)

        # 3. 바이패스 vs 시뮬레이션 전환
        if energy < self.epsilon:
            invariant = SymbolicInvariant(
                vector=I_t,
                magnitude=float(np.linalg.norm(I_t)),
                resonance_state=True,
                friction_energy=energy,
            )
            return invariant, EngineStatus.RESONANCE_BYPASS
        else:
            # 4. 내적 시뮬레이션을 통한 최소 작용 렌즈 재정렬
            delta_S = self.least_action_simulation(X_filtered, V_t)
            self.S_t = self.S_t + delta_S

            # 갱신된 렌즈축 기반 직교 재분해
            new_I_t, new_V_t, new_energy = self.compute_friction(X_filtered, self.S_t)

            invariant = SymbolicInvariant(
                vector=new_I_t,
                magnitude=float(np.linalg.norm(new_I_t)),
                resonance_state=False,
                friction_energy=new_energy,
            )
            return invariant, EngineStatus.INTERNAL_SIMULATION
