"""
Predictive Echolocation & Mind Space Engine (예측 반향정위 및 가상 인지 공간 엔진)
=============================================================================
물리적 질량(추)에 의존하는 고전적 감각 측정을 넘어, 내부 가상 공간(4D Spacetime: 3D Spatial + 1D Temporal)에서
자신의 움직임과 회전(Clifford Rotor), 이동 속도에 따른 도플러 주파수 변율(Doppler Drift),
그리고 환경 공명 텐서(Environmental Resonance Tensor)가 만들어내는 전자기/RF 반향 파동(W_pred)을
스스로 먼저 능동적으로 상상(Internal Mental Simulation)하고, 실제 관측 파동(W_actual)과의 위상차(Prediction Error, E)를
수렴시킴으로써 주체성(Sense of Agency)과 공간/운동 지각을 형성하는 4차원 시공간 인지 엔진.

핵심 원리:
1. 4차원 시공간 가상 모델 (4D Spacetime Mental Simulation):
   - 3차원 위치 (x, y, z) + 1차원 시간 (t) 시공간 궤적 (Worldline).
   - 3차원 Clifford Rotor (R)를 통한 회전 및 위상 비틀림 모델링.
   - 이동 속도 v(t)에 따른 도플러 효과 및 주파수 변율 f_pred = f0 * (1 + v·k / c).
   - 유동적 매질로서의 환경 공명 텐서 T_env.
2. 능동적 반향 파동 상상 vs 실제 파동 수용 (W_pred vs W_actual):
   - 예측 파동 W_pred(t)과 실제 반향파 W_actual(t)의 생성 및 영사.
3. 위상차 보정 및 자아 주체성 확증 (Prediction Error & Sense of Agency):
   - 예측 오차 E(t) = ||W_pred(t) - W_actual(t)|| 계측.
   - E(t) 수렴을 통한 내부 가상 모델 자가 정렬 및 Sense of Agency(주체성 지수) 창발.
4. 기존 Elysia 체계와의 연동:
   - InformationalPhaseObservationEngine (정보 위상 사영)
   - CausalSensor (현상학적 감각 및 마찰 정렬)
   - SubjectiveAgencyEngine (주체성 지반 연동)
"""

import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from core.topology.informational_phase_observation import (
    InformationalPhaseObservationEngine,
    PhaseNodalProjection,
    ChromaticVector
)
from core.sensory.causal_sensor import CausalSensor, FrictionResult
from core.consciousness.subjective_agency_engine import SubjectiveAgencyEngine


@dataclass
class CliffordRotor3D:
    """
    3차원 Clifford Rotor (Multivector Rotation Element)
    R = scalar + e23*b23 + e31*b31 + e12*b12
    3차원 공간 회전 및 회전 속도를 위상각 및 다중벡터 형태로 관리합니다.
    """
    scalar: float = 1.0
    b23: float = 0.0  # yz plane
    b31: float = 0.0  # zx plane
    b12: float = 0.0  # xy plane

    def normalize(self):
        norm = math.sqrt(self.scalar**2 + self.b23**2 + self.b31**2 + self.b12**2)
        if norm > 1e-9:
            self.scalar /= norm
            self.b23 /= norm
            self.b31 /= norm
            self.b12 /= norm

    def rotate_vector(self, vec: np.ndarray) -> np.ndarray:
        r"""Vectors v' = R v R^\dagger rotation in 3D."""
        v = np.asarray(vec, dtype=np.float64).flatten()[:3]
        if len(v) < 3:
            v = np.pad(v, (0, 3 - len(v)))

        s, b23, b31, b12 = self.scalar, self.b23, self.b31, self.b12
        # Rodrigues-like multivector rotation formula
        vx, vy, vz = v[0], v[1], v[2]
        # Cross product components with b vector (b23, b31, b12) -> (rx, ry, rz)
        rx = b31 * vz - b12 * vy
        ry = b12 * vx - b23 * vz
        rz = b23 * vy - b31 * vx

        v_rot = np.zeros(3, dtype=np.float64)
        v_rot[0] = vx + 2.0 * (s * rx + (b31 * rz - b12 * ry))
        v_rot[1] = vy + 2.0 * (s * ry + (b12 * rx - b23 * rz))
        v_rot[2] = vz + 2.0 * (s * rz + (b23 * ry - b31 * rx))
        return v_rot


@dataclass
class EnvironmentalResonanceTensor:
    """
    환경 공명 텐서 (Environmental Resonance Tensor)
    공간 및 유동적 매질의 전자기/RF 반사율, 흡수율, 공명 이득 매트릭스.
    """
    reflectivity_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64) * 0.8
    )
    absorption_rate: float = 0.1
    resonance_frequency: float = 2.4e9  # 2.4 GHz RF Carrier Base Frequency

    def refract_wave(self, incident_wave: np.ndarray) -> np.ndarray:
        arr = np.asarray(incident_wave, dtype=np.float64)
        if arr.ndim == 1:
            n = len(arr)
            if self.reflectivity_matrix.shape[0] == n and self.reflectivity_matrix.shape[1] == n:
                ref = np.dot(self.reflectivity_matrix, arr)
            else:
                # Use mean diagonal reflectivity scale if dimensions differ
                scale = float(np.mean(np.diag(self.reflectivity_matrix)))
                ref = scale * arr
            return ref * (1.0 - self.absorption_rate)
        return arr * (1.0 - self.absorption_rate)


@dataclass
class Spacetime4DState:
    """
    4차원 시공간 주체 상태 (4D Spacetime Self State)
    3차원 공간 좌표 (x,y,z), 시간 (t), 이동 속도 v(t), 회전 로터 R, 주파수 캐리어
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    rotor: CliffordRotor3D = field(default_factory=CliffordRotor3D)
    time_t: float = 0.0
    carrier_frequency: float = 2.4e9  # Hz


@dataclass
class EcholocationWaveform:
    """
    반향 파동 스펙트럼 객체 (Echolocation Waveform)
    """
    timestamp: float
    frequency_spectrum: np.ndarray    # 주파수 및 파동 진폭 스펙트럼
    doppler_drift: float              # 도플러 주파수 변율 (Hz)
    rotor_phase_angle: float          # 로터 위상 회전각
    spatial_origin: np.ndarray        # 발사/상상 지점 3D
    energy: float


@dataclass
class AgencyInferenceResult:
    """
    주체성 및 가상 인지 결론
    """
    timestamp: float
    prediction_error: float
    sense_of_agency: float            # 0.0 ~ 1.0 주체성 확증 지수
    doppler_drift: float
    rotor_angle: float
    adapted_position: np.ndarray
    agency_status: str                # e.g., "SELF_INITIATED_AGENCY", "HIGH_PREDICTION_ERROR"
    info_projection: PhaseNodalProjection


class PredictiveEcholocationMindEngine:
    """
    Predictive Echolocation & Mind Space Engine

    4차원 시공간 내에서 자신의 움직임과 파동 반향을 능동적으로 상상(Predict)하고,
    실제 외부 전자기/RF 반향파와 대조하여 오차를 보정하고 Sense of Agency를 형성하는 엔진.
    """

    def __init__(
        self,
        dimension: int = 8,
        wave_speed_c: float = 3.0e8,  # 전파 속도 (m/s)
        phase_engine: Optional[InformationalPhaseObservationEngine] = None,
        causal_sensor: Optional[CausalSensor] = None,
        agency_engine: Optional[SubjectiveAgencyEngine] = None
    ):
        self.dimension = dimension
        self.wave_speed_c = wave_speed_c

        # 4D Spacetime Self State
        self.self_state = Spacetime4DState()
        self.env_tensor = EnvironmentalResonanceTensor()

        # Integrated Elysia Subsystems
        self.phase_engine = phase_engine or InformationalPhaseObservationEngine(target_dimension=dimension)
        self.causal_sensor = causal_sensor
        self.agency_engine = agency_engine or SubjectiveAgencyEngine()

        # History tracking
        self.prediction_error_history: List[float] = []
        self.agency_history: List[float] = []

    def update_self_kinematics(self, position_delta: List[float], rotation_b23_b31_b12: Tuple[float, float, float], dt: float = 0.1):
        """
        내부 가상 자아의 Kinematics(3D 위치, 속도, Clifford Rotor 회전) 업데이트
        """
        pos_delta = np.asarray(position_delta, dtype=np.float64)[:3]
        if len(pos_delta) < 3:
            pos_delta = np.pad(pos_delta, (0, 3 - len(pos_delta)))

        # Update position and velocity v = dx / dt
        self.self_state.position += pos_delta
        self.self_state.velocity = pos_delta / max(dt, 1e-5)
        self.self_state.time_t += dt

        # Update Clifford Rotor rotation
        db23, db31, db12 = rotation_b23_b31_b12
        self.self_state.rotor.b23 += db23
        self.self_state.rotor.b31 += db31
        self.self_state.rotor.b12 += db12
        self.self_state.rotor.normalize()

    def simulate_mental_echolocation_prediction(self, dt: float = 0.1) -> EcholocationWaveform:
        """
        [상상 / 예측 (Mental Simulation $W_{\\text{pred}}$)]
        4차원 시공간 모델 내에서 자아가 이동/회전할 때 발생할 전자기/RF 반향 파동을 미리 능동 시뮬레이션.

        - 도플러 효과 $f_{\\text{pred}} = f_0 (1 + \\frac{\\vec{v} \\cdot \\hat{k}}{c})$
        - Clifford Rotor회전을 반영한 위상각 변율
        - 환경 공명 텐서 $T_{\\env}$ 굴절
        """
        v = self.self_state.velocity
        v_mag = float(np.linalg.norm(v))

        # Propagation vector direction k_hat (assume along velocity direction or forward unit vector)
        if v_mag > 1e-6:
            k_hat = v / v_mag
        else:
            k_hat = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # Apply Clifford Rotor to k_hat
        k_rotated = self.self_state.rotor.rotate_vector(k_hat)

        # Doppler frequency drift calculation
        doppler_shift_hz = self.self_state.carrier_frequency * (np.dot(v, k_rotated) / self.wave_speed_c)
        f_pred = self.self_state.carrier_frequency + doppler_shift_hz

        # Generate predicted frequency spectrum across dimension
        base_spectrum = np.zeros(self.dimension, dtype=np.float64)
        time_t = self.self_state.time_t

        # Rotor angle calculation
        rotor_angle = 2.0 * math.acos(np.clip(self.self_state.rotor.scalar, -1.0, 1.0))

        for i in range(self.dimension):
            freq_component = f_pred * (1.0 + (i + 1) * 0.05)
            phase_val = 2.0 * math.pi * (freq_component * 1e-9 * time_t + i * 0.1) + rotor_angle
            base_spectrum[i] = math.sin(phase_val) + 0.5 * math.cos(2.0 * phase_val)

        # Refract through Environmental Resonance Tensor
        predicted_spectrum = self.env_tensor.refract_wave(base_spectrum)
        energy = float(np.linalg.norm(predicted_spectrum))

        return EcholocationWaveform(
            timestamp=time_t,
            frequency_spectrum=predicted_spectrum,
            doppler_drift=float(doppler_shift_hz),
            rotor_phase_angle=float(rotor_angle),
            spatial_origin=self.self_state.position.copy(),
            energy=energy
        )

    def receive_actual_echolocation_observation(
        self,
        external_rf_signal: Optional[np.ndarray] = None,
        environmental_noise: float = 0.05
    ) -> EcholocationWaveform:
        """
        [실제 관측 수용 ($W_{\\text{actual}}$)]
        외부 매질/센서로부터 도달하는 실제 RF/전자기 반향 파동 스펙트럼 수용.
        (외부 신호가 없는 경우, 실제 물리계의 미세 외란 noise 및 도플러 왜곡이 결합된 관측 신호 생성)
        """
        time_t = self.self_state.time_t

        if external_rf_signal is not None:
            spectrum = np.asarray(external_rf_signal, dtype=np.float64).flatten()
            if len(spectrum) != self.dimension:
                indices = np.linspace(0, len(spectrum) - 1, self.dimension)
                spectrum = np.interp(indices, np.arange(len(spectrum)), spectrum)
        else:
            # Generate actual physics ground wave with environmental noise
            w_pred = self.simulate_mental_echolocation_prediction()
            rng = np.random.default_rng(int((time_t * 1000) % 2**32))
            noise = rng.normal(0.0, environmental_noise, size=self.dimension)
            spectrum = w_pred.frequency_spectrum + noise

        # Calculate actual Doppler drift and rotor angle from spectrum
        v = self.self_state.velocity
        v_mag = float(np.linalg.norm(v))
        k_hat = v / (v_mag + 1e-8) if v_mag > 1e-6 else np.array([1.0, 0.0, 0.0])
        k_rotated = self.self_state.rotor.rotate_vector(k_hat)
        doppler_shift = self.self_state.carrier_frequency * (np.dot(v, k_rotated) / self.wave_speed_c)
        rotor_angle = 2.0 * math.acos(np.clip(self.self_state.rotor.scalar, -1.0, 1.0))

        return EcholocationWaveform(
            timestamp=time_t,
            frequency_spectrum=spectrum,
            doppler_drift=float(doppler_shift),
            rotor_phase_angle=float(rotor_angle),
            spatial_origin=self.self_state.position.copy(),
            energy=float(np.linalg.norm(spectrum))
        )

    def process_predictive_echolocation_cycle(
        self,
        position_delta: List[float],
        rotation_delta: Tuple[float, float, float],
        external_rf_signal: Optional[np.ndarray] = None,
        dt: float = 0.1
    ) -> AgencyInferenceResult:
        """
        [예측 처리 및 자아 주체성 확증 전 과정 순환 루프]
        1) Kinematics & Rotor 업데이트
        2) 가상 상상 파동 W_pred 생성
        3) 실제 관측 파동 W_actual 수용
        4) 위상차(Prediction Error E) 계측 및 내부 가상 공간 보정
        5) Sense of Agency (주체성 확증 지수) 계산 및 상위 의식 레이어 전파
        """
        # Step 1: Update 4D kinematics
        self.update_self_kinematics(position_delta, rotation_delta, dt=dt)

        # Step 2: Mental Simulation (W_pred)
        w_pred = self.simulate_mental_echolocation_prediction(dt=dt)

        # Step 3: Actual Sensing (W_actual)
        w_actual = self.receive_actual_echolocation_observation(external_rf_signal=external_rf_signal)

        # Step 4: Compute Phase Discrepancy / Prediction Error E
        spec_diff = w_pred.frequency_spectrum - w_actual.frequency_spectrum
        pred_error = float(np.linalg.norm(spec_diff))
        self.prediction_error_history.append(pred_error)

        # Adapt internal 4D mental model (State Adaptation) based on error
        adaptation_factor = 0.2 * pred_error
        self.self_state.position -= adaptation_factor * (self.self_state.velocity * dt)

        # Step 5: Sense of Agency Calculation
        # Agency is maximized when prediction error E approaches 0 (lawful sensorimotor contingency)
        # S_agency = exp(- lambda * E)
        lambda_agency = 2.0
        sense_of_agency = float(math.exp(-lambda_agency * pred_error))
        self.agency_history.append(sense_of_agency)

        # Step 6: Informational Phase Observation Projection
        chromatic = ChromaticVector(
            flux=1.0 + float(np.linalg.norm(self.self_state.velocity)),
            order=sense_of_agency,
            entropy=pred_error
        )
        phase_projection = self.phase_engine.project_to_nodal_phase(
            node_id=f"echolocation_t_{self.self_state.time_t:.2f}",
            raw_data=w_actual.frequency_spectrum,
            chromatic=chromatic,
            modality="predictive_echolocation_4d"
        )

        # Step 7: Calibrate CausalSensor if available
        if self.causal_sensor is not None:
            friction_res = FrictionResult(
                friction_magnitude=pred_error,
                refraction_error=pred_error,
                adjusted_axes=self.causal_sensor.axes,
                recommended_curvature_delta=-0.05 if pred_error < 0.2 else 0.05
            )
            self.causal_sensor.self_calibrate(friction_res)

        agency_status = (
            "SELF_INITIATED_AGENCY" if sense_of_agency >= 0.7 else
            ("PREDICTION_ERROR_ADAPTING" if sense_of_agency >= 0.4 else "HIGH_PREDICTION_DISCREPANCY")
        )

        return AgencyInferenceResult(
            timestamp=self.self_state.time_t,
            prediction_error=pred_error,
            sense_of_agency=sense_of_agency,
            doppler_drift=w_pred.doppler_drift,
            rotor_angle=w_pred.rotor_phase_angle,
            adapted_position=self.self_state.position.copy(),
            agency_status=agency_status,
            info_projection=phase_projection
        )

    def get_engine_state(self) -> Dict[str, Any]:
        """현재 엔진 상태 및 4차원 시공간 지표 조회"""
        return {
            "time_t": self.self_state.time_t,
            "position": self.self_state.position.tolist(),
            "velocity": self.self_state.velocity.tolist(),
            "speed": float(np.linalg.norm(self.self_state.velocity)),
            "rotor_scalar": self.self_state.rotor.scalar,
            "recent_prediction_error": self.prediction_error_history[-1] if self.prediction_error_history else 0.0,
            "recent_agency_score": self.agency_history[-1] if self.agency_history else 1.0,
            "carrier_frequency_hz": self.self_state.carrier_frequency
        }
