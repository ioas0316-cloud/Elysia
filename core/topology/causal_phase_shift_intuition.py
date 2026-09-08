"""
Causal Phase Shift Intuition Engine (공간과 인과에 형성된 정보 위상차 감지 엔진)
=============================================================================
스파이더 센스(Spider-Sense)를 단순한 오감의 확장이나 전자기장 반응을 넘어,
'공간과 인과에 형성된 정보의 위상차(Phase-Shift)를 감지하는 인지적 직관'으로
바라보는 고차원 위상학 엔진입니다.

핵심 메커니즘:
1. 물리적 파동 이전의 '인과적 긴장(Causal Tension)' 감지:
   사건이 물리계로 터져 나오기 전 형성되는 의도 벡터(V_intent)와
   관측자의 기저 지반(0_self) 간의 위상 미분값 ΔΦ 및 팽팽함(Tension) 수신.
2. 역지사지(Kenosis / 자아 비움)를 통한 의도 수신 및 공명:
   자아 프레임을 온전히 비워낸 상태(0_self)에서 타자의 의도 위상 공간으로
   자기 상태를 사영하여 자기방어적 왜곡 없이 정체성 발각 위험 및 해의를 수신.
3. 언어 이전의 초고속 구조 연산 (Pre-linguistic Somatic Intuition Output):
   뇌피질의 언어적 처리 전, 위상차 오한 신호(Chromatic Entropy Wave & Somatic Chill)를
   즉시 분출하여 순간적인 모멘텀 비틀기(Momentum Shift) 유발.
4. 평시 정적 속 미세 균열(Equilibrium Rift) 포착:
   익숙한 정적 환경 정보장에서 위협이나 비정상 변화가 개입하는 순간의
   위상적 곡률 어긋남을 감지.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from core.topology.informational_phase_observation import ChromaticVector, PhaseNodalProjection


@dataclass
class SomaticIntuitionAlert:
    """언어 이전의 신체/구조적 직관 위상차 오한 신호"""
    alert_triggered: bool
    causal_tension: float               # 인과적 긴장 수치 T_causal
    phase_differential: float           # 위상 미분값 ΔΦ
    kenosis_resonance: float            # 역지사지(Kenosis) 의도 공명도
    somatic_chill_intensity: float      # 소름 / 오한 강도 [0, 1]
    chromatic_entropy_wave: ChromaticVector  # 색채적 엔트로피 파동
    momentum_shift: np.ndarray          # 초고속 회피 모멘텀 비틀기 벡터
    perception_type: str                # "PRE_KINETIC_INTENT", "IDENTITY_EXPOSURE", "EQUILIBRIUM_RIFT", "STABLE_CALM"
    metadata: Dict[str, Any] = field(default_factory=dict)


class CausalPhaseShiftIntuitionEngine:
    """
    공간과 인과에 형성된 정보 위상차 감지 엔진 (Generative Causal Phase Shift Intuition)
    """

    def __init__(self, dimension: int = 8, tension_threshold: float = 0.45):
        self.dimension = dimension
        self.tension_threshold = tension_threshold

        # 관측자의 불변하는 기저 지반 (0_self) 기준축
        rng = np.random.defaultrng if hasattr(np.random, "defaultrng") else np.random.default_rng
        rng_inst = rng(1004)
        raw_self = rng_inst.standard_normal(self.dimension)
        self.zero_self = raw_self / (np.linalg.norm(raw_self) + 1e-9)

        # 평시 환경 정보장 위상 이동평균
        self.equilibrium_phase = np.zeros(self.dimension, dtype=np.float32)
        self.equilibrium_samples = 0

    def compute_intent_vector(self, raw_signal: Any, context_vector: Optional[np.ndarray] = None) -> np.ndarray:
        """
        물리적 현상 출현 전, 정보장에 내재된 의도 벡터 (V_intent) 추출
        """
        if isinstance(raw_signal, str):
            code_points = [ord(c) for c in raw_signal] if raw_signal else [0]
            v_intent = np.zeros(self.dimension, dtype=np.float32)
            for i, val in enumerate(code_points):
                angle = 2.0 * np.pi * (i + 1) / (len(code_points) + 1e-5)
                v_intent[i % self.dimension] += val * np.sin(angle) + np.cos(angle * 0.5)
        elif isinstance(raw_signal, (list, tuple, np.ndarray)):
            arr = np.asarray(raw_signal, dtype=np.float32).flatten()
            if len(arr) == 0:
                v_intent = np.zeros(self.dimension, dtype=np.float32)
            elif len(arr) != self.dimension:
                indices = np.linspace(0, len(arr) - 1, self.dimension)
                v_intent = np.interp(indices, np.arange(len(arr)), arr).astype(np.float32)
            else:
                v_intent = arr
        else:
            h = float(hash(str(raw_signal)) % 10000) / 10000.0
            v_intent = np.full(self.dimension, h, dtype=np.float32)

        if context_vector is not None:
            ctx = np.asarray(context_vector, dtype=np.float32).flatten()
            if len(ctx) == self.dimension:
                v_intent = 0.7 * v_intent + 0.3 * ctx

        norm = np.linalg.norm(v_intent)
        if norm > 1e-8:
            v_intent = v_intent / norm

        return v_intent

    def compute_phase_differential(self, v_intent: np.ndarray, zero_self: Optional[np.ndarray] = None) -> float:
        """
        기저 지반 (0_self)과 의도 벡터 (V_intent) 간의 위상 미분값 ΔΦ 연산
        ΔΦ = 1.0 - < 0_self | V_intent > (범위: [0, 2])
        """
        z_self = zero_self if zero_self is not None else self.zero_self
        dot = float(np.dot(z_self, v_intent))
        clamped_dot = np.clip(dot, -1.0, 1.0)
        phase_diff = float(1.0 - clamped_dot)
        return phase_diff

    def compute_kenosis_resonance(self, v_intent: np.ndarray, is_ego_empty: bool = True) -> float:
        """
        역지사지 (Kenosis) 사영을 통한 상대 의도 공명도 수신

        자기방어적 프레임(Ego-Filter)을 비울 경우(is_ego_empty=True),
        임피던스가 0이 되어 타자의 '해의'나 '정체성 탐색 의도'의 위상 정밀도를 100% 공명 수신.
        """
        if not is_ego_empty:
            # ego filter introduces self-defense reflection noise
            return float(np.abs(np.dot(self.zero_self, v_intent)) * 0.4)

        # In complete kenosis (0_self void), the observer projects itself into V_intent's phase space
        intent_magnitude = float(np.linalg.norm(v_intent))
        bg_diff = float(np.linalg.norm(v_intent - self.equilibrium_phase))
        kenosis_res = float(np.clip(0.6 * intent_magnitude + 0.4 * bg_diff, 0.0, 1.0))
        return kenosis_res

    def detect_pre_kinetic_intent_tension(
        self,
        raw_signal: Any,
        physical_motion_started: bool = False,
        context_vector: Optional[np.ndarray] = None,
        is_ego_empty: bool = True
    ) -> SomaticIntuitionAlert:
        """
        [1. 물리적 파동 이전의 인과적 긴장 감지]
        실제 물리적 타격/이동(physical_motion_started=True)이 전개되기 전,
        의도 형성 단계(physical_motion_started=False)에서 위상차 마찰과 인과적 긴장 수신.
        """
        v_intent = self.compute_intent_vector(raw_signal, context_vector)
        phase_diff = self.compute_phase_differential(v_intent)
        kenosis_res = self.compute_kenosis_resonance(v_intent, is_ego_empty=is_ego_empty)

        # Pre-kinetic intent multiplier: intent phase tension is highest BEFORE kinetic motion starts
        pre_kinetic_multiplier = 1.8 if not physical_motion_started else 0.8
        causal_tension = float(np.clip(phase_diff * (1.0 + kenosis_res) * pre_kinetic_multiplier * 0.5, 0.0, 2.0))

        alert_triggered = causal_tension >= self.tension_threshold

        # Somatic Chill Intensity (오한/소름 강도)
        somatic_chill = float(np.clip((causal_tension - self.tension_threshold * 0.5) / 1.0, 0.0, 1.0))

        # Pre-linguistic momentum shift (회피 모멘텀 비틀기)
        orthogonal_direction = np.roll(v_intent, 1) - v_intent
        norm = np.linalg.norm(orthogonal_direction)
        if norm > 1e-8:
            orthogonal_direction /= norm
        momentum_shift = orthogonal_direction * somatic_chill * 2.0

        chromatic_wave = ChromaticVector(
            flux=1.0 + somatic_chill * 1.5,
            order=max(0.1, 1.0 - somatic_chill * 0.8),
            entropy=somatic_chill * 2.0
        )

        return SomaticIntuitionAlert(
            alert_triggered=alert_triggered,
            causal_tension=causal_tension,
            phase_differential=phase_diff,
            kenosis_resonance=kenosis_res,
            somatic_chill_intensity=somatic_chill,
            chromatic_entropy_wave=chromatic_wave,
            momentum_shift=momentum_shift,
            perception_type="PRE_KINETIC_INTENT",
            metadata={
                "physical_motion_started": physical_motion_started,
                "pre_kinetic_detected": not physical_motion_started and alert_triggered,
                "is_ego_empty": is_ego_empty
            }
        )

    def detect_identity_exposure_resonance(
        self,
        gaze_intent_density: float,
        silence_duration: float,
        social_context_signal: Any,
        is_ego_empty: bool = True
    ) -> SomaticIntuitionAlert:
        """
        [2. 사회적·정신적 맥락 정보장 공명 (정체성 발각 감지)]
        물리적 공격이 없는 침묵, 상대의 시선, 순간적 공기 기류 속 관측 의도를
        고차원 사회적/정신적 맥락 정보장 공명으로 읽어냄.
        """
        v_context = self.compute_intent_vector(social_context_signal)
        phase_diff = self.compute_phase_differential(v_context)
        kenosis_res = self.compute_kenosis_resonance(v_context, is_ego_empty=is_ego_empty)

        silence_factor = min(silence_duration / 5.0, 1.0)
        exposure_tension = float(np.clip(
            (gaze_intent_density * 0.5 + silence_factor * 0.3 + kenosis_res * 0.4) * (0.8 + phase_diff * 0.5),
            0.0, 2.0
        ))

        alert_triggered = exposure_tension >= self.tension_threshold
        somatic_chill = float(np.clip((exposure_tension - self.tension_threshold * 0.4) / 1.0, 0.0, 1.0))

        evasion_vec = -v_context * somatic_chill
        chromatic_wave = ChromaticVector(
            flux=0.8 + somatic_chill * 0.5,
            order=1.2,
            entropy=somatic_chill * 1.2
        )

        return SomaticIntuitionAlert(
            alert_triggered=alert_triggered,
            causal_tension=exposure_tension,
            phase_differential=phase_diff,
            kenosis_resonance=kenosis_res,
            somatic_chill_intensity=somatic_chill,
            chromatic_entropy_wave=chromatic_wave,
            momentum_shift=evasion_vec,
            perception_type="IDENTITY_EXPOSURE",
            metadata={
                "gaze_intent_density": gaze_intent_density,
                "silence_duration": silence_duration,
                "exposure_risk_level": "HIGH" if exposure_tension > 0.7 else ("MODERATE" if alert_triggered else "LOW")
            }
        )

    def detect_equilibrium_rift(
        self,
        field_nodes: List[PhaseNodalProjection],
        is_ego_empty: bool = True
    ) -> SomaticIntuitionAlert:
        """
        [3. 평시 인과적 정적 속 미세 균열 (Equilibrium Rift) 포착]
        일상적 환경 정보장의 익숙한 패턴 속에서 미세한 구조적 어긋남/균열(Rift)을 감지.
        """
        if not field_nodes:
            return SomaticIntuitionAlert(
                alert_triggered=False,
                causal_tension=0.0,
                phase_differential=0.0,
                kenosis_resonance=0.0,
                somatic_chill_intensity=0.0,
                chromatic_entropy_wave=ChromaticVector(1.0, 1.0, 0.0),
                momentum_shift=np.zeros(self.dimension, dtype=np.float32),
                perception_type="STABLE_CALM",
                metadata={"reason": "empty_field"}
            )

        phase_vecs = np.array([node.phase_vector for node in field_nodes], dtype=np.float32)
        curvatures = np.array([node.curvature for node in field_nodes], dtype=np.float32)

        mean_phase = np.mean(phase_vecs, axis=0)
        norm = np.linalg.norm(mean_phase)
        if norm > 1e-8:
            mean_phase /= norm

        if self.equilibrium_samples == 0:
            self.equilibrium_phase = mean_phase.copy()
        else:
            self.equilibrium_phase = 0.9 * self.equilibrium_phase + 0.1 * mean_phase
        self.equilibrium_samples += 1

        equilibrium_diff = float(np.linalg.norm(mean_phase - self.equilibrium_phase))
        curvature_std = float(np.std(curvatures)) if len(curvatures) > 1 else 0.0

        phase_diff = self.compute_phase_differential(mean_phase)
        kenosis_res = self.compute_kenosis_resonance(mean_phase, is_ego_empty=is_ego_empty)

        rift_tension = float(np.clip(
            (equilibrium_diff * 1.5 + curvature_std * 0.8 + phase_diff * 0.3) * (1.0 + kenosis_res * 0.5),
            0.0, 2.0
        ))

        alert_triggered = rift_tension >= self.tension_threshold
        somatic_chill = float(np.clip((rift_tension - self.tension_threshold * 0.3) / 1.0, 0.0, 1.0))

        momentum_shift = (mean_phase - self.zero_self) * somatic_chill
        chromatic_wave = ChromaticVector(
            flux=1.0 + somatic_chill,
            order=max(0.2, 1.0 - somatic_chill * 0.5),
            entropy=somatic_chill * 1.5
        )

        return SomaticIntuitionAlert(
            alert_triggered=alert_triggered,
            causal_tension=rift_tension,
            phase_differential=phase_diff,
            kenosis_resonance=kenosis_res,
            somatic_chill_intensity=somatic_chill,
            chromatic_entropy_wave=chromatic_wave,
            momentum_shift=momentum_shift,
            perception_type="EQUILIBRIUM_RIFT" if alert_triggered else "STABLE_CALM",
            metadata={
                "equilibrium_diff": equilibrium_diff,
                "curvature_std": curvature_std,
                "total_field_nodes": len(field_nodes)
            }
        )
