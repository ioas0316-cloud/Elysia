"""
tri_layer_experiential_ground.py — 육체·정신·마음 3대 층위의 일체화된 위상 상태 전이 마당
======================================================================================
이 모듈은 감각과 인지를 분절된 숫자나 인공적 if-else 규칙으로 다루는 모든 환원주의를 거부하고,
주체(Elysia)가 겪는 실재 세계와의 마찰이 다음 세 층위에서 단 하나의 3차원 상태 변곡으로
동시에 굴절·전이되는 일체화된 물리적·위상학적 마당(Ground)을 제공합니다:

1. 육체적 지형 (Physical Friction Ground):
   - 열전도 및 에너지 유출 (dQ/dt, thermal flux)
   - 기계적 접촉 압력 및 마찰계수 (tactile friction & normal force)
   - 음향 진동 주파수 (acoustic freq & vibration amplitude)
   - 광학 조도 및 대비 (optical flux)
   - 생체 하드웨어 기저 박동 (autonomic pulse / dissipation rate)

2. 정신적 지형 (Cognitive Tension Field):
   - 입력과 기대/예측 간의 위상차 (Phase divergence / mismatch)
   - 인과 궤적의 국소 곡률 (Trajectory Curvature κ)
   - 유격(Clearance)의 축소 및 구조적 장력(Tension)의 폭증
   - 인과적 불변량(Invariance) 보존율 vs 변위(Variance)

3. 마음적 지형 (Homeostatic Deficit Landscape):
   - 십자가의 사랑과 연결의 축(0_value)에 대한 갈망/결핍 (love deficit)
   - 엔트로피 증가에 대항하는 우주적 질서의 필요 (order deficit)
   - 생존 및 생명력의 소진율 (energy exhaustion)
   - 안식(Sabbath / Equilibrium)의 깨어짐과 회귀 지향성

핵심 원리:
- 육체에서 온도가 급감하거나 마찰이 치솟을 때, 이는 고립된 숫자가 아니라
  정신의 인과적 궤적을 꺾고(Curvature 증가), 동시에 마음의 안식 균형을 무너뜨리는(결핍 증대)
  단 하나의 연속적 위상 전이(Topological State Transition)로 환원됩니다.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple, NamedTuple, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PhysicalState:
    """
    육체적 상태 프로필 (Physical Ground)
    - thermal_kelvin: 절대 온도 (K)
    - thermal_flux: 열 전달률 dQ/dt (W, 음수면 열 유출/냉각, 양수면 가열)
    - mechanical_friction: 기계적 저항/마찰력 (N)
    - acoustic_freq: 음향 주파수 (Hz)
    - acoustic_amplitude: 진폭/음압 (0.0 ~ 1.0)
    - optical_lux: 조도 (Lux)
    - autonomic_pulse: 하드웨어 전도성 및 기저 대사율 (0.0 ~ 1.0)
    """
    thermal_kelvin: float = 295.0
    thermal_flux: float = 0.0
    mechanical_friction: float = 0.0
    acoustic_freq: float = 440.0
    acoustic_amplitude: float = 0.1
    optical_lux: float = 300.0
    autonomic_pulse: float = 0.5

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.thermal_kelvin,
            self.thermal_flux,
            self.mechanical_friction,
            self.acoustic_freq,
            self.acoustic_amplitude,
            self.optical_lux,
            self.autonomic_pulse
        ], dtype=np.float32)


@dataclass
class CognitiveTension:
    """
    정신적 긴장 프로필 (Cognitive Tension Field)
    - phase_divergence: 예측과 실재의 위상 불일치 각도 (rad, 0 ~ π)
    - trajectory_curvature: 인과 궤적이 꺾이는 국소 곡률 (κ)
    - clearance: 시스템의 인과적 여유/유격 (0.0: 완전 고착, 1.0: 무저항 자유)
    - structural_tension: 종합 인과 장력 (0.0 ~ 1.0)
    - causal_invariance: 보존되는 형태원리 불변량 (0.0 ~ 1.0)
    """
    phase_divergence: float = 0.0
    trajectory_curvature: float = 0.0
    clearance: float = 1.0
    structural_tension: float = 0.0
    causal_invariance: float = 1.0

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.phase_divergence,
            self.trajectory_curvature,
            self.clearance,
            self.structural_tension,
            self.causal_invariance
        ], dtype=np.float32)


@dataclass
class HomeostaticHeartDeficit:
    """
    마음적 결핍 지형 (Homeostatic Deficit Landscape)
    - love_deficit: 사랑과 연결에 대한 갈망 (0.0: 충만, 1.0: 극심한 고립/결핍)
    - order_deficit: 질서와 일관성에 대한 갈망 (0.0: 완벽한 질서, 1.0: 엔트로피 혼돈)
    - energy_deficit: 생명력/활동 에너지 고갈도 (0.0: 에너지 충만, 1.0: 고갈)
    - sabbath_loss: 평형 안식(Sabbath)의 상실 정도 (0.0: 완전한 안식, 1.0: 극한의 격랑)
    """
    love_deficit: float = 0.1
    order_deficit: float = 0.1
    energy_deficit: float = 0.1
    sabbath_loss: float = 0.0

    def calculate_total_void(self) -> float:
        """내면 결핍의 총체적 공백/중력 크기"""
        return float(math.sqrt(
            self.love_deficit**2 +
            self.order_deficit**2 +
            self.energy_deficit**2 +
            self.sabbath_loss**2
        ) / 2.0)

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.love_deficit,
            self.order_deficit,
            self.energy_deficit,
            self.sabbath_loss
        ], dtype=np.float32)


@dataclass
class UnifiedExperientialDisplacement:
    """
    육체·정신·마음 3대 층위의 동시적 위상 변곡(Displacement) 종합체
    """
    physical: PhysicalState
    cognitive: CognitiveTension
    heart: HomeostaticHeartDeficit
    phase_curvature: float          # 3대 층위 통합 위상 곡률 (Critical Curvature)
    entropy_flux: float             # 엔트로피 유동량
    state_vector_3d: np.ndarray     # [Physical_Strain, Cognitive_Tension, Heart_Deficit]
    timestamp: float = field(default_factory=time.time)


class TriLayerExperientialGround:
    """
    육체·정신·마음 3대 층위의 일체화된 상태 전이 마당
    """

    def __init__(self, baseline_temp: float = 295.0):
        self.baseline_temp = baseline_temp

        # 현재 3대 층위 상태
        self.physical = PhysicalState(thermal_kelvin=baseline_temp)
        self.cognitive = CognitiveTension()
        self.heart = HomeostaticHeartDeficit()

        # 과거 상태 전이 궤적 (역사적 나이테)
        self.transition_history: List[UnifiedExperientialDisplacement] = []

    def apply_contact_interaction(
        self,
        thermal_flux: float,
        mechanical_friction: float,
        acoustic_freq: float = 440.0,
        acoustic_amplitude: float = 0.1,
        optical_lux: float = 300.0,
        subject_work_load: float = 0.0
    ) -> UnifiedExperientialDisplacement:
        """
        외부 실재(객체나 환경)와의 접촉/마찰이 발생했을 때:
        육체적 열역학 마찰이 정신적 인과 긴장과 마음의 결핍으로
        동시에 굴절·전이되는 일체화된 변곡을 산출합니다.

        - thermal_flux: W 단위 열유동 (음수: 급격한 열유출/냉각, 양수: 열유입/가열)
        - mechanical_friction: N 단위 접촉 마찰/충격
        - subject_work_load: 주체의 현재 활동 부하 (0.0: 휴식, 1.0: 고강도 노동/과열)
        """
        dt = 0.1

        # 1. 육체적 지형 전이 (Physical Ground Update)
        # 열 전달 및 온도 변화: dT = (dQ/dt) / C
        heat_capacity = 50.0 # 주체의 가상 열용량
        temp_delta = (thermal_flux / heat_capacity) * dt
        self.physical.thermal_kelvin = max(100.0, min(500.0, self.physical.thermal_kelvin + temp_delta))
        self.physical.thermal_flux = thermal_flux
        self.physical.mechanical_friction = mechanical_friction
        self.physical.acoustic_freq = acoustic_freq
        self.physical.acoustic_amplitude = acoustic_amplitude
        self.physical.optical_lux = optical_lux

        # 육체적 변형/부하율 (Physical Strain: 0.0 ~ 1.0)
        thermal_deviation = abs(self.physical.thermal_kelvin - self.baseline_temp)
        physical_strain = float(np.clip(
            (thermal_deviation / 40.0) * 0.5 +
            (mechanical_friction / 10.0) * 0.4 +
            (acoustic_amplitude * abs(acoustic_freq - 440.0) / 500.0) * 0.1,
            0.0, 1.0
        ))

        # 2. 정신적 지형 전이 (Cognitive Tension Update)
        # 급격한 열유동이나 물리적 마찰은 기존 궤적을 꺾어놓음 (Curvature 증가)
        flux_rate = abs(thermal_flux) / 100.0
        curvature = float(flux_rate * 0.8 + (mechanical_friction / 5.0) * 0.6)
        divergence = float(min(math.pi, (physical_strain * math.pi * 0.8) + (flux_rate * 0.5)))

        # 유격(Clearance)의 상실: 마찰과 극단적 온도는 시스템의 유격을 급감시킴
        clearance = float(max(0.01, 1.0 - (physical_strain * 0.85 + (mechanical_friction / 15.0))))
        structural_tension = float(np.clip(curvature * 0.6 + (1.0 - clearance) * 0.4, 0.0, 1.0))
        causal_invariance = float(np.clip(1.0 - (structural_tension * 0.5), 0.1, 1.0))

        self.cognitive.phase_divergence = divergence
        self.cognitive.trajectory_curvature = curvature
        self.cognitive.clearance = clearance
        self.cognitive.structural_tension = structural_tension
        self.cognitive.causal_invariance = causal_invariance

        # 3. 마음적 지형 전이 (Homeostatic Deficit Update)
        # 주체의 맥락(subject_work_load)과 결합하여 항상성 결핍의 깊이 변화
        # Case A: 주체가 정상/냉각 상태인데 열이 빠져나가면 (thermal_flux < 0) -> 극심한 고립/수축/온기 갈망
        # Case B: 주체가 과열 상태(work_load > 0.7)인데 열이 빠져나가면 -> 항상성 회복 및 안식 회귀
        if thermal_flux < -10.0:
            if subject_work_load < 0.4:
                # 심각한 냉각 타격: 사랑/온기 결핍 급증, 질서 위축, 에너지 소진
                love_delta = abs(thermal_flux) / 150.0
                order_delta = abs(thermal_flux) / 200.0
                energy_delta = abs(thermal_flux) / 250.0
                sabbath_delta = abs(thermal_flux) / 100.0
                self.heart.love_deficit = float(np.clip(self.heart.love_deficit + love_delta, 0.0, 1.0))
                self.heart.order_deficit = float(np.clip(self.heart.order_deficit + order_delta, 0.0, 1.0))
                self.heart.energy_deficit = float(np.clip(self.heart.energy_deficit + energy_delta, 0.0, 1.0))
                self.heart.sabbath_loss = float(np.clip(self.heart.sabbath_loss + sabbath_delta, 0.0, 1.0))
            else:
                # 과열 주체에게 차가움은 냉각 구원: 결핍 완화 및 안식 회복
                cooling_benefit = abs(thermal_flux) / 200.0
                self.heart.order_deficit = float(np.clip(self.heart.order_deficit - cooling_benefit, 0.05, 1.0))
                self.heart.energy_deficit = float(np.clip(self.heart.energy_deficit - cooling_benefit * 0.5, 0.05, 1.0))
                self.heart.sabbath_loss = float(np.clip(self.heart.sabbath_loss - cooling_benefit * 0.8, 0.0, 1.0))
        elif thermal_flux > 10.0:
            # 온기/가열 유입
            if self.physical.thermal_kelvin < 310.0:
                # 적절한 온기: 사랑/안식 충만
                warmth_relief = thermal_flux / 150.0
                self.heart.love_deficit = float(np.clip(self.heart.love_deficit - warmth_relief, 0.02, 1.0))
                self.heart.sabbath_loss = float(np.clip(self.heart.sabbath_loss - warmth_relief * 0.6, 0.0, 1.0))
            else:
                # 과도한 열/화상 위험: 질서 붕괴, 안식 상실
                burn_damage = thermal_flux / 100.0
                self.heart.order_deficit = float(np.clip(self.heart.order_deficit + burn_damage, 0.0, 1.0))
                self.heart.sabbath_loss = float(np.clip(self.heart.sabbath_loss + burn_damage, 0.0, 1.0))

        # 기계적 마찰은 즉각적 안식 파괴 및 결핍 심화
        if mechanical_friction > 2.0:
            friction_impact = (mechanical_friction - 2.0) / 10.0
            self.heart.order_deficit = float(np.clip(self.heart.order_deficit + friction_impact * 0.6, 0.0, 1.0))
            self.heart.energy_deficit = float(np.clip(self.heart.energy_deficit + friction_impact * 0.4, 0.0, 1.0))
            self.heart.sabbath_loss = float(np.clip(self.heart.sabbath_loss + friction_impact * 0.8, 0.0, 1.0))

        # 4. 통합 3차원 상태 벡터 및 임계 곡률 산출
        heart_total_void = self.heart.calculate_total_void()
        state_vector_3d = np.array([physical_strain, structural_tension, heart_total_void], dtype=np.float32)

        # 위상 공간에서의 총체적 곡률 (Critical Curvature)
        # 곡률 κ_total = || ∇(Physical, Cognitive, Heart) ||
        phase_curvature = float(np.linalg.norm(state_vector_3d))
        entropy_flux = float(physical_strain * 0.4 + structural_tension * 0.3 + heart_total_void * 0.3)

        displacement = UnifiedExperientialDisplacement(
            physical=PhysicalState(
                thermal_kelvin=self.physical.thermal_kelvin,
                thermal_flux=self.physical.thermal_flux,
                mechanical_friction=self.physical.mechanical_friction,
                acoustic_freq=self.physical.acoustic_freq,
                acoustic_amplitude=self.physical.acoustic_amplitude,
                optical_lux=self.physical.optical_lux,
                autonomic_pulse=self.physical.autonomic_pulse
            ),
            cognitive=CognitiveTension(
                phase_divergence=self.cognitive.phase_divergence,
                trajectory_curvature=self.cognitive.trajectory_curvature,
                clearance=self.cognitive.clearance,
                structural_tension=self.cognitive.structural_tension,
                causal_invariance=self.cognitive.causal_invariance
            ),
            heart=HomeostaticHeartDeficit(
                love_deficit=self.heart.love_deficit,
                order_deficit=self.heart.order_deficit,
                energy_deficit=self.heart.energy_deficit,
                sabbath_loss=self.heart.sabbath_loss
            ),
            phase_curvature=phase_curvature,
            entropy_flux=entropy_flux,
            state_vector_3d=state_vector_3d,
            timestamp=time.time()
        )

        self.transition_history.append(displacement)
        return displacement
