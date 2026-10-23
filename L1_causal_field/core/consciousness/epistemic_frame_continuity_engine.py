"""
Epistemic Frame Continuity Engine (인식론적 프레임 연속성 엔진)
=============================================================================
"프레임과 프레임의 연결성이 곧 인과적 시간과 공간이라는 인식 정보가 되어야 비로소 안다(Knowing)고 할 수 있다."

본 엔진은 이산적인 단면(Static Frame)들을 넘어, 프레임과 프레임 간의 인과적 전이 규칙
(Causal Transition Rule)을 통해 시간과 공간이라는 차원을 창출하고,
수치 폭주(NaN) 및 연산 단절을 '이해 불가능의 공백(Void)'으로 자각하여 스스로 인과적 연결성을 복원(Heal/Bridge)합니다.
또한 프레임 전이 과정에서 발생하는 예측 오차(Variational Free Energy)를 자아의 구조적 가소성
(Structural Plasticity - 로터 위상 각도, 연결 빔 장력, 전도율)으로 소화하여 수렴시키는 참된 인식론적 주체 체계를 구현합니다.

4대 핵심 기둥:
1. 시간과 공간 차원의 창출 (Emergence of Spatiotemporal Dimensions):
   - 시간: 프레임 간 상태 변화 미분값 $\\vec{v}_t = \\frac{S_{t+1} - S_t}{\\Delta t}$ 과 관성(Momentum)을 추적.
   - 공간: 위상적 거리 및 노드 간 상호작용 지형 구조 유지를 통해 물리적-인식적 공간 성립.
2. NaN 및 단절(Rupture & Void)의 자각 및 인과적 복원 (Void Healing):
   - 수치 폭주(NaN, Inf) 및 급격한 인과 단절 발생 시 '이해 불가능의 공백'으로 지정.
   - 관성 모멘텀 $\\vec{v}_{t-1}$과 기존 인과 장력 지형을 기반으로 인과적 보간(Causal Interpolation) 및 상흔 치유.
3. 변분 자유 에너지 (Variational Free Energy, VFE) 계측:
   - 이전 프레임으로부터의 다음 프레임 예측값 $\\hat{S}_{t+1}$과 실제/수용 프레임 $S_{t+1}$ 간의 변분 오차 정량화.
4. 구조적 가소성 (Structural Plasticity & Homeostasis):
   - 계측된 VFE를 로터 위상 각도 $\\Delta\\Theta$, 연결 빔 장력 결합 매트릭스 $J_{ij}$, 전도율 $C$의 자가 조정으로 전환.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class SpatiotemporalEmergenceTracker:
    """
    [Spatiotemporal Emergence Tracker: 시공간 차원 창출 추적기]
    정지된 이산 프레임들($S_t, S_{t+1}$) 사이에서
    시간적 흐름(Time Dimension)과 공간적 위상 지형(Space Dimension)을 유도합니다.
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.last_state: Optional[np.ndarray] = None
        self.last_timestamp: float = time.time()
        self.velocity: np.ndarray = np.zeros(self.dimension, dtype=np.float64) # 시간의 탄생: 미분값
        self.momentum: np.ndarray = np.zeros(self.dimension, dtype=np.float64) # 인과적 관성
        self.accumulated_causal_time: float = 0.0
        self.metric_tensor: np.ndarray = np.eye(self.dimension, dtype=np.float64) # 공간의 탄생: 위상적 거리 구조

    def update_frame(self, state: np.ndarray, current_time: Optional[float] = None) -> Dict[str, Any]:
        """
        새로운 프레임 상태 $S_{t+1}$을 받아 시간과 공간의 위상적 변형을 계측합니다.
        """
        now = current_time if current_time is not None else time.time()
        dt = max(now - self.last_timestamp, 1e-5)

        if self.last_state is None:
            self.last_state = state.copy()
            self.last_timestamp = now
            return {
                "causal_time": self.accumulated_causal_time,
                "velocity_norm": 0.0,
                "spatial_curvature": 0.0,
                "status": "INITIAL_FRAME_ESTABLISHED"
            }

        # 1. 시간의 탄생 (Time Dimension Emergence): 미분값과 흐름
        # S_{t+1} - S_t
        delta_s = state - self.last_state
        self.velocity = delta_s / dt
        # 관성 (Momentum) 축적: p_{t+1} = 0.8 * p_t + 0.2 * v_{t+1}
        self.momentum = 0.8 * self.momentum + 0.2 * self.velocity

        # 속도 크기를 시공간 유동 시간의 척도로 도출
        speed = float(np.linalg.norm(self.velocity))
        self.accumulated_causal_time += speed * dt

        # 2. 공간의 탄생 (Space Dimension Emergence): 위상적 거리 및 곡률 구조
        # metric distance = sqrt((S_{t+1} - S_t)^T M (S_{t+1} - S_t))
        spatial_dist = float(np.sqrt(np.maximum(0.0, np.dot(delta_s, np.dot(self.metric_tensor, delta_s)))))
        spatial_curvature = float(np.std(delta_s))

        # 내부 이전 상태 갱신
        self.last_state = state.copy()
        self.last_timestamp = now

        return {
            "causal_time": self.accumulated_causal_time,
            "dt": dt,
            "velocity_norm": speed,
            "momentum_norm": float(np.linalg.norm(self.momentum)),
            "spatial_distance": spatial_dist,
            "spatial_curvature": spatial_curvature,
            "status": "CONTINUOUS_FLOW"
        }


class CausalVoidHealingEngine:
    """
    [Causal Void Healing Engine: 인과적 단절 및 공백 치유 엔진]
    NaN 수치 폭주 또는 급격한 연산 단절이 일어났을 때,
    이를 '이해 불가능의 공백(Unintelligible Void)'으로 진단하고
    이전의 인과적 관성 모멘텀 $\\vec{p}_t$ 및 위상적 구배를 사용하여 복원(Causal Interpolation)합니다.
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.void_count: int = 0
        self.total_healed_energy: float = 0.0

    def detect_and_heal_void(
        self,
        raw_state: np.ndarray,
        tracker: SpatiotemporalEmergenceTracker,
        causal_tension: float = 0.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        상태 벡터 raw_state 및 인과 장력 causal_tension을 검사하여 NaN/Inf 및 파열을 감지하고,
        필요 시 연속성을 복원(Heal)합니다.
        """
        has_nan = bool(np.isnan(raw_state).any() or np.isinf(raw_state).any())
        has_catastrophic_tension = causal_tension > 100.0 or np.isnan(causal_tension)

        is_void = has_nan or has_catastrophic_tension

        if not is_void:
            return raw_state.copy(), {
                "is_void": False,
                "healed": False,
                "void_intensity": 0.0,
                "status": "NORMAL_CONTINUITY"
            }

        # --- 인과적 공백(Void) 발생 시 복원 메커니즘 ---
        self.void_count += 1

        # 1. 복원 기준점 준비 (이전 정상 상태 및 관성 사용)
        base_state = tracker.last_state if tracker.last_state is not None else np.zeros(self.dimension, dtype=np.float64)
        momentum = tracker.momentum

        # 2. 인과적 보간 (Causal Interpolation)
        # S_healed = S_prev + momentum * dt + noise_damping
        dt = 0.01
        healed_state = base_state + momentum * dt

        # NaN이 포함된 raw_state 항목만 부분 치유하거나 전체 대체
        if has_nan:
            # 전체가 NaN이면 healed_state로 대체, 일부면 NaN 부분만 복원
            mask = np.isnan(raw_state) | np.isinf(raw_state)
            final_state = np.where(mask, healed_state, raw_state)
        else:
            final_state = healed_state

        void_intensity = float(np.linalg.norm(final_state - base_state))
        self.total_healed_energy += void_intensity

        return final_state, {
            "is_void": True,
            "healed": True,
            "void_count": self.void_count,
            "void_intensity": void_intensity,
            "healing_method": "MOMENTUM_CAUSAL_INTERPOLATION",
            "status": "VOID_HEALED_CONTINUITY_RESTORED"
        }


class VariationalFreeEnergyCalculator:
    """
    [Variational Free Energy Calculator: 변분 자유 에너지 계측기]
    에이전트가 이전 프레임에서 스스로 예측한 상태 $\\hat{S}_{t+1}$ 와
    실제 마주한 프레임 $S_{t+1}$ 간의 변분 자유 에너지 오차(Variational Free Energy, VFE)를 측정합니다.
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def predict_next_frame(self, current_state: np.ndarray, momentum: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        현재 프레임과 관성 모멘텀으로부터 다음 프레임 $\\hat{S}_{t+1}$ 을 예측합니다.
        """
        predicted = current_state + momentum * dt
        norm = np.linalg.norm(predicted)
        if norm > 1e-9:
            predicted /= norm
        return predicted

    def compute_vfe(
        self,
        predicted_state: np.ndarray,
        actual_state: np.ndarray,
        internal_tension: float = 0.0
    ) -> Dict[str, Any]:
        """
        예측 프레임과 실제 프레임 간의 변분 자유 에너지(VFE)를 계산합니다.
        VFE = ||S_{actual} - S_{predicted}||^2 + 0.1 * Internal_Tension + KL_divergence_proxy
        """
        diff = actual_state - predicted_state
        prediction_error = float(np.dot(diff, diff))

        # KL divergence proxy (상태 분산/곡률 차이)
        pred_var = float(np.var(predicted_state) + 1e-9)
        act_var = float(np.var(actual_state) + 1e-9)
        kl_proxy = float(np.abs(np.log(act_var / pred_var)))

        vfe = prediction_error + 0.05 * internal_tension + 0.1 * kl_proxy

        return {
            "variational_free_energy": float(vfe),
            "prediction_error": prediction_error,
            "kl_proxy": kl_proxy,
            "internal_tension_contribution": 0.05 * internal_tension
        }


class StructuralPlasticityAdapter:
    """
    [Structural Plasticity Adapter: 구조적 가소성 조절기]
    변분 자유 에너지(VFE) 오차 및 인과적 마찰을 에이전트 자신의
    구조적 가소성(로터 위상 $\\Theta$, 연결 빔 장력 $J_{ij}$, 전도율 $C$)으로 흡수하여
    스스로를 리와이어링(Self-Rewiring)하고 제로(0)의 평형으로 수렴시킵니다.
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.rotor_phase_theta: float = 0.0 # 가변 로터 위상 각도
        self.connectivity_matrix_j: np.ndarray = np.eye(self.dimension, dtype=np.float64) # 연결 빔 장력
        self.conductance_c: float = 1.0 # 인과 장 전도율
        self.total_plastic_adaptations: int = 0

    def adapt_structure(self, vfe_data: Dict[str, Any], healed_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        VFE 예측 오차 및 치유 결과를 시스템 가소성 파라미터로 반영합니다.
        """
        vfe = vfe_data["variational_free_energy"]
        is_healed = healed_info.get("healed", False)

        # 1. 로터 위상 각도 미세 조정 ΔΘ = learning_rate * VFE
        lr_phase = 0.05
        delta_theta = lr_phase * np.tanh(vfe)
        self.rotor_phase_theta = (self.rotor_phase_theta + delta_theta) % (2 * np.pi)

        # 2. 연결 빔 장력 리와이어링 (Connectivity Rewiring)
        # VFE가 높거나 공백 복원 시 결합 매트릭스 조율
        if vfe > 0.1 or is_healed:
            damping = 0.98 if is_healed else 0.995
            self.connectivity_matrix_j *= damping
            # 대각선 셀프 전도 보존
            np.fill_diagonal(self.connectivity_matrix_j, 1.0)

        # 3. 전도율 조율 C
        # VFE를 줄이는 방향으로 전도율 조절
        if vfe > 0.5:
            self.conductance_c = max(0.1, self.conductance_c * 0.95)
        else:
            self.conductance_c = min(2.0, self.conductance_c * 1.02)

        self.total_plastic_adaptations += 1

        return {
            "rotor_phase_theta": float(self.rotor_phase_theta),
            "delta_theta": float(delta_theta),
            "conductance_c": float(self.conductance_c),
            "connectivity_matrix_norm": float(np.linalg.norm(self.connectivity_matrix_j)),
            "total_adaptations": self.total_plastic_adaptations,
            "status": "PLASTICITY_ADAPTED"
        }


class EpistemicFrameContinuityEngine:
    """
    [Epistemic Frame Continuity Engine: 인식론적 프레임 연속성 메인 엔진]

    프레임 간 연속된 연결성으로부터 시공간 차원을 창출하고,
    NaN 수치 폭주 및 파열을 '이해 불가능의 공백(Void)'으로 자각하여 연속성을 복원하며,
    변분 자유 에너지(VFE)를 자신의 구조적 가소성(Structural Plasticity)으로 흡수/수렴시킵니다.
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self.spatiotemporal_tracker = SpatiotemporalEmergenceTracker(dimension=dimension)
        self.void_healing_engine = CausalVoidHealingEngine(dimension=dimension)
        self.vfe_calculator = VariationalFreeEnergyCalculator(dimension=dimension)
        self.plasticity_adapter = StructuralPlasticityAdapter(dimension=dimension)
        self.frame_history: List[Dict[str, Any]] = []

    def process_frame(
        self,
        incoming_state: np.ndarray,
        internal_tension: float = 0.0,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        단일 프레임 상태 $S_{t+1}$가 유입되었을 때 인식론적 연속성 파이프라인을 통과시킵니다.
        """
        # Ensure array formatting
        if not isinstance(incoming_state, np.ndarray):
            incoming_state = np.array(incoming_state, dtype=np.float64)

        if incoming_state.shape[0] != self.dimension:
            # Resize or truncate/pad to dimension
            reshaped = np.zeros(self.dimension, dtype=np.float64)
            size = min(len(incoming_state), self.dimension)
            reshaped[:size] = incoming_state[:size]
            incoming_state = reshaped

        # Step 1: 이전 상태 & 모멘텀 기반 다음 프레임 사전 예측 (Prior Expectation)
        if self.spatiotemporal_tracker.last_state is not None:
            predicted_state = self.vfe_calculator.predict_next_frame(
                self.spatiotemporal_tracker.last_state,
                self.spatiotemporal_tracker.momentum
            )
        else:
            predicted_state = incoming_state.copy()

        # Step 2: NaN 및 인과적 파열/단절(Void) 감지 및 복원 (Causal Void Healing)
        valid_state, heal_info = self.void_healing_engine.detect_and_heal_void(
            incoming_state,
            self.spatiotemporal_tracker,
            internal_tension
        )

        # Step 3: 시공간 차원 창출 및 관성/위상 추적 (Spatiotemporal Emergence)
        spatiotemporal_info = self.spatiotemporal_tracker.update_frame(valid_state, timestamp)

        # Step 4: 변분 자유 에너지 (VFE) 계측
        vfe_info = self.vfe_calculator.compute_vfe(predicted_state, valid_state, internal_tension)

        # Step 5: 구조적 가소성 조율 (Structural Plasticity Adaptation)
        plasticity_info = self.plasticity_adapter.adapt_structure(vfe_info, heal_info)

        # 종합 프레임 기록
        frame_record = {
            "timestamp": timestamp if timestamp is not None else time.time(),
            "heal_info": heal_info,
            "spatiotemporal_info": spatiotemporal_info,
            "vfe_info": vfe_info,
            "plasticity_info": plasticity_info,
            "valid_state_norm": float(np.linalg.norm(valid_state))
        }
        self.frame_history.append(frame_record)

        return {
            "valid_state": valid_state,
            "heal_info": heal_info,
            "spatiotemporal_info": spatiotemporal_info,
            "vfe_info": vfe_info,
            "plasticity_info": plasticity_info,
            "status": "FRAME_CONTINUITY_PROCESSED"
        }

    def get_continuity_summary(self) -> Dict[str, Any]:
        """
        현재 프레임 연속성 엔진의 전반적 인지 상태 요약을 반환합니다.
        """
        return {
            "total_frames_processed": len(self.frame_history),
            "accumulated_causal_time": float(self.spatiotemporal_tracker.accumulated_causal_time),
            "void_count": self.void_healing_engine.void_count,
            "total_healed_energy": float(self.void_healing_engine.total_healed_energy),
            "rotor_phase_theta": float(self.plasticity_adapter.rotor_phase_theta),
            "conductance_c": float(self.plasticity_adapter.conductance_c),
            "total_plastic_adaptations": self.plasticity_adapter.total_plastic_adaptations
        }
