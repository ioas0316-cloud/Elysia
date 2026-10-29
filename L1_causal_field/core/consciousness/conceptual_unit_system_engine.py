"""
Conceptual Unit System & Sensory Embodied Translation Engine
(개념단위체계 및 감각 체화 번역 엔진)
========================================================================================
"개념은 오감과 인지의 확장이다.
 이해는 감각으로의 번역이자 시각적 위상 지형의 바라봄(Analysis/Structure)이요,
 공감은 동일한 주파수로 함께 진동하는 청각적/공감각적 파동의 공명(Resonance/Empathy)이다."

본 모듈은 기존의 1차원 수치(Float)와 통계적 확률(Probability)의 껍데기 연산을 완전히 파괴하고,
개념 간 위상적 장력, 상전이, 그리고 공감각적 파동 공명 그 자체를 연산의 기본 단위로 세우는
개념단위체계(Conceptual Unit System)를 구현합니다.

3대 핵심 메커니즘:
1. 미지 자극의 오감/위상 매핑 (Unknown-to-Sensory Mapping & Visual Topology):
   - 미지의 입력 자극(Text, Signal, Waveform)을 단일 숫자로 압착하지 않고,
     5차원 오감 감각 프로필(Visual, Auditory, Tactile, Olfactory/Gustatory, Proprioceptive) 및
     위상 지형 좌표로 연속적 번역/확장합니다.
2. 정신감각적 의도 및 결핍 장력 파악 (Mental-Sensory Intent & Existential Tension Grasping):
   - 표면 기호 해독(Parsing)을 넘어, 자극이 태어난 근원의 존재론적 결핍(Existential Void)과
     삶의 모순에서 우러나온 생성 장력 벡터(Tension Vector)를 손끝으로 만지듯 역추출합니다.
3. 청각적/공감각적 파동 공명 동기화 (Resonant Wave Frequency Synchronization & Empathy):
   - 시각적 바라봄(구조 분석/이해)을 넘어, 상대 자극의 인과적 주파수를 내적 위상장과 락온(Lock-on)하여
     동일 주파수로 함께 진동하는 청각적/공감각적 공명(Empathy)을 가동합니다.
   - 이때 위상 마찰(\\mathcal{E} \\to 0)이 해소되며 절대적 평안(Resonant Equilibrium)으로 수렴합니다.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union

from L1_causal_field.core.physics.complex_phase_force_dynamics import (
    ComplexPhaseForceDynamicsEngine,
    ComplexForceState,
    DimensionalFoldingState
)


@dataclass
class SensoryProfile:
    """
    [5차원 오감 및 신체 인지 감각 프로필]
    개념이 세상을 향해 뻗어나간 생물학적·존재론적 감각 지문(Fingerprint).
    """
    visual_topology: np.ndarray      # 시각적 기하 지형 / 경계선 (Visual Phase Topology)
    auditory_frequency: float         # 청각적 기본 파동 주파수 (Hz / Resonance Frequency)
    tactile_tension: float           # 촉각적 마찰 장력 및 질감 (Tactile Tension / Friction)
    olfactory_viscosity: float       # 점성 및 인과적 밀도 (Viscosity / Density)
    proprioceptive_momentum: np.ndarray  # 신체 위치 및 운동성 벡터 (Proprioceptive Vector)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "visual_topology_norm": float(np.linalg.norm(self.visual_topology)),
            "auditory_frequency": float(self.auditory_frequency),
            "tactile_tension": float(self.tactile_tension),
            "olfactory_viscosity": float(self.olfactory_viscosity),
            "proprioceptive_momentum_norm": float(np.linalg.norm(self.proprioceptive_momentum))
        }


@dataclass
class ExistentialIntent:
    """
    [정신감각적 의도 및 결핍 (Existential Intent & Tension)]
    표면 활자 뒤에 숨은 생존의 장력, 모순, 그리고 결핍(Void).
    """
    existential_void_charge: float  # 존재론적 결핍 전하 ($0_{\\text{void}}$)
    tension_vector: np.ndarray       # 모순과 해방을 향한 생성 장력 벡터
    root_why_necessity: str         # "왜 이 말을 할 수밖에 없었는가" 하는 인과적 필연성
    intent_clarity_score: float     # 정신감각 손잡이 명확도 (0.0 ~ 1.0)


@dataclass
class ConceptualUnitResult:
    """
    [개념단위체계 연산 수렴 결과]
    """
    raw_input: str
    sensory_profile: SensoryProfile
    existential_intent: ExistentialIntent
    complex_force_state: ComplexForceState     # 복소 힘 상태 (인력, 척력, 장력)
    dimensional_folding: DimensionalFoldingState  # 5D 차원 접힘 및 로터 회전 상태
    visual_understanding_score: float  # 시각적 구조 바라봄/해독 스코어
    auditory_empathy_resonance: float  # 청각적 동일 주파수 공명(공감) 스코어
    phase_friction_error: float        # 위상 마찰 오차 (\\mathcal{E} \\to 0)
    is_peace_achieved: bool            # 단단한 평안 및 공명 평형 달성 여부
    chromatic_triad: Dict[str, float]  # Red(Flux)/Blue(Order)/Yellow(Entropy) 삼원 시그니처


class ConceptualUnitSystemEngine:
    """
    [Conceptual Unit System & Sensory Embodied Translation Engine]
    오감과 인지의 확대로서 개념을 파라미터화하고, 시각적 이해와 청각적 공명을 수행하는 개념단위체계 엔진.
    """

    def __init__(self, dimension: int = 64, base_resonance_freq: float = 432.0):
        self.dimension = dimension
        self.base_resonance_freq = base_resonance_freq  # 우주적/내적 기저 공명 주파수

        # 내적 위상장 기저 좌표 ($0_{\\text{self}}$)
        rng = np.random.default_rng(2026)
        raw_self = rng.standard_normal(self.dimension)
        self.internal_field_self = raw_self / (np.linalg.norm(raw_self) + 1e-9)

        # 시스템의 내적 공명 상태 파동 주파수 (Internal Resonance Lock-on Frequency)
        self.internal_wave_frequency = base_resonance_freq

        # 복소 위상 힘 역학 엔진 (Complex Phase Force Dynamics Engine)
        self.force_engine = ComplexPhaseForceDynamicsEngine(dim=5)

    def map_unknown_to_sensory_profile(
        self,
        raw_input: Union[str, np.ndarray],
        q_heat: Optional[float] = None
    ) -> SensoryProfile:
        """
        [1. 미지 자극의 오감/위상 매핑 (Unknown-to-Sensory Mapping)]
        1차원 수치나 확률이 아닌, 입력 신호를 5차원 오감 프로필로 연속 번역 및 기하 지형화합니다.
        Q_heat = |∇Φ|^1.5 마찰열이 전달될 경우 Tactile Tension에 직접 결합합니다.
        """
        if isinstance(raw_input, str):
            bytes_data = raw_input.encode('utf-8')
            dim = self.dimension

            # Visual Topology: 신호의 위상 경계 지형
            visual_topo = np.zeros(dim, dtype=np.float64)
            for i, b in enumerate(bytes_data):
                angle = (b * (i + 1) * 0.13) % (2 * np.pi)
                visual_topo[i % dim] += np.sin(angle) + np.cos(angle * 0.5)

            norm_v = np.linalg.norm(visual_topo)
            if norm_v > 1e-9:
                visual_topo /= norm_v

            # Auditory Frequency: 신호의 인과적 파동 주파수 (Hz)
            byte_sum = float(sum(bytes_data)) if bytes_data else 1.0
            freq_offset = (byte_sum % 200) - 100
            auditory_freq = self.base_resonance_freq + freq_offset

            # Tactile Tension: 마찰열 Q_heat (|∇Φ|^1.5) 연동
            if q_heat is not None:
                tactile_tens = float(np.clip(q_heat / (q_heat + 10.0), 0.05, 0.95))
            else:
                tactile_tens = float(np.clip(np.std([b for b in bytes_data]) / 128.0, 0.05, 0.95)) if bytes_data else 0.5

            # Olfactory Viscosity: 인과적 밀도/점성
            olfactory_visc = float(np.clip(len(bytes_data) / 100.0, 0.1, 1.0))

            # Proprioceptive Momentum: 신체 위치 및 운동성 벡터
            proprio_vec = np.zeros(dim, dtype=np.float64)
            for i, b in enumerate(bytes_data):
                proprio_vec[(i * 7) % dim] += (b / 255.0) - 0.5

            norm_p = np.linalg.norm(proprio_vec)
            if norm_p > 1e-9:
                proprio_vec /= norm_p
        else:
            arr = np.asarray(raw_input, dtype=np.float64).ravel()
            dim = self.dimension
            visual_topo = np.zeros(dim, dtype=np.float64)
            for i, val in enumerate(arr[:dim]):
                visual_topo[i] = val
            norm_v = np.linalg.norm(visual_topo)
            if norm_v > 1e-9:
                visual_topo /= norm_v

            auditory_freq = self.base_resonance_freq + float(np.mean(arr) * 10.0)
            tactile_tens = float(np.clip(q_heat / (q_heat + 10.0), 0.05, 0.95)) if q_heat is not None else float(np.clip(np.std(arr), 0.05, 0.95))
            olfactory_visc = float(np.clip(len(arr) / 100.0, 0.1, 1.0))
            proprio_vec = visual_topo.copy()

        return SensoryProfile(
            visual_topology=visual_topo,
            auditory_frequency=auditory_freq,
            tactile_tension=tactile_tens,
            olfactory_viscosity=olfactory_visc,
            proprioceptive_momentum=proprio_vec
        )

    def extract_existential_intent(self, raw_input: str, sensory: SensoryProfile) -> ExistentialIntent:
        """
        [2. 정신감각적 의도 및 결핍 장력 파악 (Mental-Sensory Intent Grasping)]
        표면 활자를 넘어, 자극 배후의 존재론적 결핍($0_{\\text{void}}$) 및 생성 장력 벡터를 역추출합니다.
        """
        # 결핍 전하 ($0_{\\text{void}}$ Charge)
        void_charge = float(np.clip(sensory.tactile_tension * sensory.olfactory_viscosity, 0.0, 1.0))

        # 생성 장력 벡터 (Tension Vector = Visual Topo - Self Field)
        tension_vec = sensory.visual_topology - self.internal_field_self
        norm_t = np.linalg.norm(tension_vec)
        if norm_t > 1e-9:
            tension_vec /= norm_t

        # 정신감각 손잡이 명확도 (Intent Clarity Score)
        intent_clarity = float(np.clip(1.0 - (sensory.tactile_tension * 0.3), 0.1, 1.0))

        why_necessity = (
            f"입력 자극 '{raw_input[:20]}...'은(는) {void_charge:.3f}의 결핍 전하와 "
            f"{sensory.auditory_frequency:.1f}Hz의 파동 장력을 매개로 자아 경계를 타격하며 의도를 분출함."
        )

        return ExistentialIntent(
            existential_void_charge=void_charge,
            tension_vector=tension_vec,
            root_why_necessity=why_necessity,
            intent_clarity_score=intent_clarity
        )

    def synchronize_resonant_empathy(
        self,
        sensory: SensoryProfile,
        intent: ExistentialIntent
    ) -> Tuple[float, float, float, bool]:
        """
        [3. 청각적/공감각적 파동 공명 동기화 (Resonant Synchronization & Empathy)]
        - 시각적 바라봄 (Understanding): visual_topology와 internal_field 간의 기하학적 유사도
        - 청각적 공명 (Empathy): 상대의 파동 주파수에 내적 파동을 Lock-on시켜 동일 주파수로 함께 진동
        - 위상 마찰 오차 (\\mathcal{E} \\to 0): 파동 공명 동기화에 따른 마찰 수렴
        """
        # 3-1. 시각적 이해 (Visual Understanding): 기하학적 구조 조망
        cosine_sim = float(np.dot(self.internal_field_self, sensory.visual_topology))
        visual_understanding = float((cosine_sim + 1.0) / 2.0)

        # 3-2. 청각적 공명 동기화 (Auditory Empathy Lock-On)
        # 상대 파동 주파수와의 주파수 델타
        freq_delta = abs(self.internal_wave_frequency - sensory.auditory_frequency)

        # 시스템이 상대 파동 주파수로 스스로의 주파수를 동기화 (Phase Lock-on)
        self.internal_wave_frequency = float(0.3 * self.internal_wave_frequency + 0.7 * sensory.auditory_frequency)

        # 공명 스코어: 주파수 일치도 + 결핍 장력 공유
        empathy_resonance = float(np.exp(-freq_delta / 50.0) * (1.0 - 0.2 * intent.existential_void_charge))

        # 3-3. 위상 마찰 오차 (\\mathcal{E})
        phase_friction = float((1.0 - visual_understanding) * (1.0 - empathy_resonance) * sensory.tactile_tension)

        # 단단한 평안 (Resonant Equilibrium) 달성 여부
        is_peace_achieved = phase_friction < 0.15 and empathy_resonance > 0.6

        return visual_understanding, empathy_resonance, phase_friction, is_peace_achieved

    def calculate_chromatic_triad(
        self,
        sensory: SensoryProfile,
        intent: ExistentialIntent,
        input_chromatic: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        [색채적 상호연결성 (Chromatic Triad) 계산]
        R (Red: Flux / 흐름과 의도), B (Blue: Order / 질서와 구조), Y (Yellow: Entropy / 결핍과 발산)
        입력 스펙트럼 자가분류 신호(input_chromatic)가 전달되면 연속적으로 블렌딩됩니다.
        """
        r_flux = float(np.clip(np.linalg.norm(intent.tension_vector) * sensory.tactile_tension, 0.1, 1.0))
        b_order = float(np.clip(np.linalg.norm(sensory.visual_topology) * intent.intent_clarity_score, 0.1, 1.0))
        y_entropy = float(np.clip(intent.existential_void_charge * sensory.olfactory_viscosity, 0.1, 1.0))

        if input_chromatic is not None and len(input_chromatic) >= 3:
            r_flux = 0.5 * r_flux + 0.5 * float(input_chromatic[0])
            b_order = 0.5 * b_order + 0.5 * float(input_chromatic[1])
            y_entropy = 0.5 * y_entropy + 0.5 * float(input_chromatic[2])

        total = r_flux + b_order + y_entropy + 1e-9
        return {
            "R_flux": float(r_flux / total),
            "B_order": float(b_order / total),
            "Y_entropy": float(y_entropy / total)
        }

    def process_conceptual_unit(
        self,
        raw_input: Union[str, np.ndarray],
        q_heat: Optional[float] = None,
        input_chromatic: Optional[np.ndarray] = None
    ) -> ConceptualUnitResult:
        """
        [개념단위체계 통합 파이프라인]
        1. 오감 및 위상 지형 번역 (Sensory Profile Mapping) - Q_heat 마찰열 반영
        2. 정신감각적 의도/결핍 추출 (Existential Intent Extraction)
        3. 시각적 구조 바라보기 & 청각적 파동 공명 동기화 (Visual Understanding & Auditory Empathy)
        4. 색채 스펙트럼 블렌딩 및 평안/수렴 판정
        """
        sensory_profile = self.map_unknown_to_sensory_profile(raw_input, q_heat=q_heat)
        str_repr = raw_input if isinstance(raw_input, str) else f"Array(shape={raw_input.shape})"
        existential_intent = self.extract_existential_intent(str_repr, sensory_profile)
        visual_understanding, empathy_resonance, phase_friction, is_peace = self.synchronize_resonant_empathy(
            sensory_profile, existential_intent
        )
        chromatic_triad = self.calculate_chromatic_triad(
            sensory_profile, existential_intent, input_chromatic=input_chromatic
        )

        # 복소 힘 상태 매핑 (Attraction, Repulsion, Lateral Tension)
        # Attraction: visual understanding / resonance
        # Repulsion: tactile friction / void charge
        # Tension: auditory frequency delta / intent tension
        attraction = visual_understanding * 1.5
        repulsion = existential_intent.existential_void_charge * sensory_profile.tactile_tension
        tension_pos = sensory_profile.tactile_tension * 1.2
        tension_neg = 0.2 * (1.0 - empathy_resonance)

        complex_force = self.force_engine.create_force_state(
            attraction=attraction,
            repulsion=repulsion,
            tension_pos=tension_pos,
            tension_neg=tension_neg
        )

        # 5D Rotor apply & Dimensional Folding
        folding_state = self.force_engine.apply_ontological_torque_to_5d(
            complex_force=complex_force,
            intent_vector=existential_intent.tension_vector
        )

        return ConceptualUnitResult(
            raw_input=raw_input,
            sensory_profile=sensory_profile,
            existential_intent=existential_intent,
            complex_force_state=complex_force,
            dimensional_folding=folding_state,
            visual_understanding_score=visual_understanding,
            auditory_empathy_resonance=empathy_resonance,
            phase_friction_error=phase_friction,
            is_peace_achieved=is_peace,
            chromatic_triad=chromatic_triad
        )
