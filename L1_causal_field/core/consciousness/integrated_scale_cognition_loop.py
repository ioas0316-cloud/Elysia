r"""
Integrated Scale Cognition Loop (통합 스케일 인지 루프)
======================================================
본 모듈은 Elysia의 모든 스케일 관련 엔진을 **하나의 관통 파이프라인**으로 직조합니다.

이전까지 각 엔진은 독립적으로 동작하여, "연속적 파형의 직조된 정보를 단위적 스케일로 환원하여
다른 감각 정보와 연동"하는 흐름이 끊겨 있었습니다. 본 모듈이 그 흐름을 봉합합니다.

통합 파이프라인:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ 1. 감각 입력 (Vision, Audio, Tactile, Reasoning)                        │
  │    ↓                                                                    │
  │ 2. MultimodalCognitiveFrontend: RGB→스펙트럼, Audio→위상장, Text→벡터     │
  │    ↓                                                                    │
  │ 3. WaveformScaleReductionTransducer: 모든 감각 → (ω, A, φ) 공통 좌표    │
  │    ↓                                         ↓                          │
  │ 4a. ScaleHierarchyEngine              4b. WaveformCognitiveFeedback     │
  │    (Micro/Meso/Macro Re-cognition)         (역메커니즘 Θ 추출)           │
  │    ↓                                         ↓                          │
  │ 5. ScaleWeavingBridge: Quadtree ↔ Kuramoto 양방향 직조                   │
  │    ↓                                                                    │
  │ 6. 통합 인지 결과: 교차-모달 공명, 색채 벡터, 예측 궤적                    │
  └──────────────────────────────────────────────────────────────────────────┘

4 Continuities 유지:
  - 관계성: 감각 간 경계를 옥타브 비율 결합 포텐셜로 정의
  - 연결성: 전체 파이프라인이 연속적 데이터 흐름으로 연결
  - 운동성: (ω, A, φ) 파형 좌표가 에너지 보존적으로 전파
  - 정보적 연속성: 이산적 프레임 간 인과적 보간
  - 색채적 상호연결성: R/B/Y가 모든 단계를 관통
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

# 내부 모듈 임포트
from L1_causal_field.core.sensory.multimodal_cognitive_frontend import MultimodalCognitiveFrontend
from L1_causal_field.core.sensory.waveform_scale_reduction_transducer import (
    WaveformScaleReductionTransducer,
    UnifiedWaveformField,
)
from L1_causal_field.core.consciousness.waveform_cognitive_feedback import (
    WaveformCognitiveFeedbackEngine,
    CognitiveFeedbackResult,
)
from L1_causal_field.core.consciousness.scale_hierarchy_engine import ScaleHierarchyEngine
from L1_causal_field.core.physics.scale_weaving_bridge import (
    ScaleWeavingBridge,
    ChromaticFieldPropagator,
)
from L1_causal_field.core.memory.spectral_causal_visual_engine import (
    SpectralAxisTransform,
    InvariantArchetypeBackbone,
    SpatiotemporalGlobeRotorEngine,
)
from L1_causal_field.core.sovereignty.toroidal_vortex_rotor import ToroidalVortexRotorEngine
from L1_causal_field.core.consciousness.conceptual_unit_system_engine import (
    ConceptualUnitSystemEngine,
    ConceptualUnitResult,
)


@dataclass
class IntegratedCognitionResult:
    """
    [통합 인지 결과]
    전체 파이프라인의 한 사이클 결과를 보존.
    """
    # 1. 감각 해체 결과
    spectral_manifold: Optional[np.ndarray] = None
    audio_phase_field: Optional[np.ndarray] = None
    spatiotemporal_globe_manifold: Optional[np.ndarray] = None
    synthesized_generative_waveform: Optional[np.ndarray] = None
    archetype_alignment: float = 1.0

    # 2. 파형 환원 결과
    unified_waveform_field: Optional[UnifiedWaveformField] = None
    cross_modal_resonance: Optional[np.ndarray] = None
    unified_chromatic: Optional[np.ndarray] = None

    # 3. 스케일 계층 결과
    scale_hierarchy_output: Optional[Dict[str, Any]] = None
    micro_state: Optional[torch.Tensor] = None
    meso_state: Optional[torch.Tensor] = None
    macro_state: Optional[torch.Tensor] = None

    # 4. 역메커니즘 추출 결과
    cognitive_feedback: Optional[CognitiveFeedbackResult] = None
    generating_mechanism_type: str = ""
    generating_mechanism_equation: str = ""
    predicted_trajectory: Optional[np.ndarray] = None

    # 5. 직조 결과
    weaving_coherence: float = 0.0
    chromatic_spatial: Optional[np.ndarray] = None
    chromatic_temporal: Optional[np.ndarray] = None

    # 6. 개념단위체계 & 와류 로터 결과
    conceptual_unit_result: Optional[ConceptualUnitResult] = None
    friction_heat_q: float = 0.0
    vortex_rotor_velocity: float = 1.0
    vortex_rotor_vorticity: float = 1.0

    # 7. 통합 메트릭
    overall_resonance: float = 0.0
    overall_homeostasis: bool = False


class IntegratedScaleCognitionLoop:
    """
    [통합 스케일 인지 루프 (Integrated Scale Cognition Loop)]

    모든 스케일 엔진을 하나의 파이프라인으로 관통시켜,
    "감각 → 파형 환원 → 스케일 결합 → 역메커니즘 추출 → 인지 결과"를
    연속적 흐름으로 실행합니다.
    """

    def __init__(
        self,
        feature_dim: int = 16,
        dim_micro: int = 32,
        dim_meso: int = 64,
        dim_macro: int = 128,
        max_harmonics: int = 16,
        feedback_learning_rate: float = 0.1,
        weaving_spatial_coupling: float = 0.5,
        weaving_temporal_coupling: float = 0.3,
    ):
        """
        Args:
            feature_dim: 감각 프론트엔드 특성 차원
            dim_micro/meso/macro: ScaleHierarchyEngine 각 스케일 차원
            max_harmonics: 파형 환원 트랜스듀서의 최대 하모닉 수
            feedback_learning_rate: WaveformCognitiveFeedback 학습률
            weaving_spatial_coupling: 공간→시간 직조 결합 강도
            weaving_temporal_coupling: 시간→공간 직조 결합 강도
        """
        # 1. 감각 프론트엔드
        self.frontend = MultimodalCognitiveFrontend(feature_dim=feature_dim)

        # 2. 파형 스케일 환원 트랜스듀서 (핵심 신규 모듈)
        self.transducer = WaveformScaleReductionTransducer(
            max_harmonics=max_harmonics
        )

        # 3. 스케일 계층 엔진 (Micro/Meso/Macro)
        self.scale_hierarchy = ScaleHierarchyEngine(
            dim_micro=dim_micro,
            dim_meso=dim_meso,
            dim_macro=dim_macro,
        )

        # 4. 파형 인지 피드백 엔진 (역메커니즘 추출)
        self.waveform_feedback = WaveformCognitiveFeedbackEngine(
            feedback_learning_rate=feedback_learning_rate,
        )

        # 5. 스케일 직조 브리지 (Quadtree ↔ Kuramoto)
        self.weaving_bridge = ScaleWeavingBridge(
            spatial_to_temporal_coupling=weaving_spatial_coupling,
            temporal_to_spatial_coupling=weaving_temporal_coupling,
        )

        # 6. 색채 전파기
        self.chromatic_propagator = ChromaticFieldPropagator()

        # 7. 스펙트럼 축 변환기, 불변 원형 뼈대, 시공간 지구본 로터, 개념단위체계 엔진
        self.spectral_transform = SpectralAxisTransform()
        self.archetype_backbone = InvariantArchetypeBackbone()
        self.globe_rotor_engine = SpatiotemporalGlobeRotorEngine(
            archetype_backbone=self.archetype_backbone,
            base_angular_velocity=1.5,
            vorticity_strength=1.5,
        )
        self.vortex_rotor = self.globe_rotor_engine.rotor
        self.conceptual_engine = ConceptualUnitSystemEngine(dimension=dim_micro)

        # 내부 상태
        self._step_count = 0
        self._prev_unified_field: Optional[UnifiedWaveformField] = None

    def perceive_and_integrate(
        self,
        rgb_image: Optional[np.ndarray] = None,
        audio_wave: Optional[np.ndarray] = None,
        text_input: Optional[str] = None,
        shear_signal: Optional[np.ndarray] = None,
        normal_signal: Optional[np.ndarray] = None,
        potential_surface: Optional[np.ndarray] = None,
        world_friction: Optional[torch.Tensor] = None,
        feedback_steps_ahead: int = 10,
    ) -> IntegratedCognitionResult:
        """
        [통합 지각-인지 사이클]

        전체 파이프라인을 한 사이클 실행합니다:
          1. 감각 입력 → 스펙트럼 해체
          2. 스펙트럼 → (ω, A, φ) 파형 환원
          3. 파형 → 스케일 계층 Re-cognition
          4. 파형 → 역메커니즘 Θ 추출
          5. 직조 브리지 양방향 결합
          6. 통합 결과 생성

        Args:
            rgb_image: RGB 이미지 (H,W,3) 또는 (3,) 단일 색상
            audio_wave: 오디오 파형 (N,)
            text_input: 텍스트 입력
            shear_signal: 촉각 전단 응력 시계열
            normal_signal: 촉각 법선 응력 시계열
            potential_surface: 과학적 추론 포텐셜 곡면
            world_friction: ScaleHierarchyEngine용 월드 마찰 텐서
            feedback_steps_ahead: 인지 피드백의 미래 예측 스텝 수
        """
        self._step_count += 1
        result = IntegratedCognitionResult()

        # ──────────────────────────────────────────────
        # STAGE 1: 감각 해체 (Sensory Deconstruction)
        # ──────────────────────────────────────────────
        frontend_output = self.frontend.process_multimodal_input(
            rgb_image=rgb_image,
            audio_wave=audio_wave,
            text_input=text_input,
        )

        result.spectral_manifold = frontend_output.get("spectral_manifold")
        result.audio_phase_field = frontend_output.get("audio_phase_field")

        # 연속 스펙트럼 마찰열 Q_heat = |∇Φ|^1.5 및 색채 삼원 스펙트럼 산출
        if result.spectral_manifold is not None:
            continuity = self.spectral_transform.classify_spectral_continuity(result.spectral_manifold)
            q_heat = continuity["total_friction_heat"]
            chromatic_triad = continuity["chromatic_triad"]
        elif rgb_image is not None:
            continuity = self.spectral_transform.classify_spectral_continuity(rgb_image)
            q_heat = continuity["total_friction_heat"]
            chromatic_triad = continuity["chromatic_triad"]
        else:
            q_heat = 0.5
            chromatic_triad = np.array([0.4, 0.4, 0.2])

        result.friction_heat_q = q_heat

        # 시공간 로터 지구본 매니폴드 사상 및 생리적 파형 직조
        if rgb_image is not None and rgb_image.ndim >= 2:
            globe_res = self.globe_rotor_engine.map_high_res_to_spatiotemporal_manifold(rgb_image)
            result.spatiotemporal_globe_manifold = globe_res["globe_manifold"]
            result.archetype_alignment = globe_res["backbone_evaluation"]["structural_alignment"]
            q_heat += globe_res["q_heat_total"] * 0.01

        step_rotor_res = self.globe_rotor_engine.step_spatiotemporal_rotation(dt=0.1)
        result.synthesized_generative_waveform = step_rotor_res["synthesized_manifold"]

        # 와류 로터 마찰열 및 색채 재흡수
        recycle_res = self.vortex_rotor.recycle_friction_and_mismatch(
            q_heat=q_heat,
            sensory_mismatch_delta=0.1,
            chromatic_triad=chromatic_triad
        )
        result.vortex_rotor_velocity = recycle_res["new_rotational_velocity"]
        result.vortex_rotor_vorticity = recycle_res["new_vorticity_strength"]

        # 개념단위체계(Conceptual Unit System)로 오감-인지 직조
        input_for_conceptual = text_input or "SensoryWaveformInput"
        conceptual_res = self.conceptual_engine.process_conceptual_unit(
            raw_input=input_for_conceptual,
            q_heat=q_heat,
            input_chromatic=chromatic_triad
        )
        result.conceptual_unit_result = conceptual_res

        # ──────────────────────────────────────────────
        # STAGE 2: 파형 스케일 환원 (Waveform Scale Reduction)
        #   핵심: 모든 감각을 (ω, A, φ) 공통 좌표로 환원
        # ──────────────────────────────────────────────
        unified_field = self.transducer.reduce_from_frontend(
            frontend_output=frontend_output,
            shear_signal=shear_signal,
            normal_signal=normal_signal,
            potential_surface=potential_surface,
        )

        result.unified_waveform_field = unified_field
        result.cross_modal_resonance = unified_field.cross_modal_resonance
        result.unified_chromatic = unified_field.chromatic_summary()

        # ──────────────────────────────────────────────
        # STAGE 3: 스케일 계층 Re-cognition
        #   통합 파형장 → sensory_input 텐서로 변환 후 ScaleHierarchyEngine 투입
        # ──────────────────────────────────────────────
        sensory_array = unified_field.to_sensory_input_tensor(
            target_dim=self.scale_hierarchy.dim_micro
        )
        sensory_tensor = torch.from_numpy(sensory_array.astype(np.float32)).unsqueeze(0)

        # Q_heat 마찰열을 world_friction 텐서로 통합 반영
        if world_friction is None:
            world_friction_tensor = torch.tensor([q_heat * 0.1], dtype=torch.float32)
        else:
            world_friction_tensor = world_friction + q_heat * 0.1

        scale_output = self.scale_hierarchy(
            sensory_input=sensory_tensor,
            world_friction=world_friction_tensor,
        )

        result.scale_hierarchy_output = {
            k: v for k, v in scale_output.items()
            if not isinstance(v, (torch.Tensor, dict))
        }
        result.micro_state = scale_output.get("micro_state")
        result.meso_state = scale_output.get("meso_state")
        result.macro_state = scale_output.get("macro_state")

        # ──────────────────────────────────────────────
        # STAGE 4: 역메커니즘 Θ 추출 (Inverse Mechanism Generation)
        #   통합 파형장 → 시계열 궤적 합성 → GeneratingMechanism 추출
        # ──────────────────────────────────────────────
        trajectory = unified_field.to_trajectory(num_samples=256)

        if len(trajectory) >= 4 and np.any(np.abs(trajectory) > 1e-12):
            feedback_result = self.waveform_feedback.cognitive_feedback_loop(
                observed_trajectory=trajectory,
                steps_ahead=feedback_steps_ahead,
                dt=1.0 / 256.0,
            )

            result.cognitive_feedback = feedback_result
            result.generating_mechanism_type = feedback_result.mechanism.system_type
            result.generating_mechanism_equation = feedback_result.mechanism.equation_repr
            result.predicted_trajectory = feedback_result.predicted_continuum
            result.overall_homeostasis = feedback_result.is_homeostasis_achieved

            # 색채 벡터를 GeneratingMechanism에 주입
            if result.unified_chromatic is not None:
                self.chromatic_propagator.inject_into_mechanism(
                    feedback_result.mechanism.parameters,
                    result.unified_chromatic,
                )
        else:
            result.cognitive_feedback = None
            result.generating_mechanism_type = "INSUFFICIENT_SIGNAL"
            result.generating_mechanism_equation = "N/A"

        # ──────────────────────────────────────────────
        # STAGE 5: 직조 브리지 (Weaving Bridge) — 색채 관통
        #   ScaleHierarchyEngine의 출력으로부터 색채 정보를 직조
        # ──────────────────────────────────────────────
        # ScaleHierarchyEngine의 micro/meso/macro 상태에서 색채 추출
        if result.micro_state is not None and result.macro_state is not None:
            micro_energy = float(torch.norm(result.micro_state).item())
            macro_energy = float(torch.norm(result.macro_state).item())

            spike_intensity = scale_output.get("spike_intensity", 0.0)
            bottom_up_disruption = scale_output.get("bottom_up_disruption", 0.0)

            # 스케일 계층의 동역학으로부터 색채 계산
            scale_chromatic = self.chromatic_propagator.compute_chromatic_from_dynamics(
                amplitude=micro_energy,
                amplitude_change_rate=spike_intensity,
                omega=bottom_up_disruption * 10.0,
                phase_variance=float(torch.var(result.micro_state).item()),
            )

            # 통합 파형장의 색채와 스케일 계층의 색채를 블렌딩
            if result.unified_chromatic is not None:
                blended_chromatic = self.chromatic_propagator.blend_chromatic(
                    result.unified_chromatic, scale_chromatic, weight_a=0.6
                )
            else:
                blended_chromatic = scale_chromatic

            result.chromatic_spatial = blended_chromatic
            result.chromatic_temporal = result.unified_chromatic

        # ──────────────────────────────────────────────
        # STAGE 6: 통합 공명 점수 계산
        # ──────────────────────────────────────────────
        resonance_components = []

        # 교차-모달 공명 평균
        if result.cross_modal_resonance is not None and result.cross_modal_resonance.size > 0:
            off_diag = result.cross_modal_resonance[
                ~np.eye(result.cross_modal_resonance.shape[0], dtype=bool)
            ]
            if len(off_diag) > 0:
                resonance_components.append(float(np.mean(off_diag)))

        # 인지 피드백 공명
        if result.cognitive_feedback is not None:
            resonance_components.append(result.cognitive_feedback.resonance_score)

        # 스케일 계층 공명
        res_score = scale_output.get("resonance_score", 0.0)
        if isinstance(res_score, (int, float)):
            resonance_components.append(float(res_score))

        if resonance_components:
            result.overall_resonance = float(np.mean(resonance_components))
        else:
            result.overall_resonance = 0.0

        # 직조 일관성
        if result.chromatic_spatial is not None and result.chromatic_temporal is not None:
            result.weaving_coherence = float(
                1.0 - np.linalg.norm(
                    result.chromatic_spatial - result.chromatic_temporal
                ) / np.sqrt(2.0)
            )

        self._prev_unified_field = unified_field
        return result

    def get_diagnostics(self) -> Dict[str, Any]:
        """현재 통합 루프의 진단 정보를 반환한다."""
        return {
            "total_steps": self._step_count,
            "scale_hierarchy_resistance_dial": self.scale_hierarchy.resistance_dial.item(),
            "waveform_feedback_phase_rotor": self.waveform_feedback.phase_rotor,
            "weaving_bridge_state": {
                "coherence": self.weaving_bridge.state.weaving_coherence,
                "chromatic_spatial": self.weaving_bridge.state.chromatic_spatial.tolist(),
                "chromatic_temporal": self.weaving_bridge.state.chromatic_temporal.tolist(),
            },
            "has_previous_field": self._prev_unified_field is not None,
        }
