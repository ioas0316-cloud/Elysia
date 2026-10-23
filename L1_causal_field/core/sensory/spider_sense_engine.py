"""
Spider Sense Engine (스파이더 센스 감각 인터페이스 모듈)
=============================================================================
`CausalPhaseShiftIntuitionEngine`을 래핑하여, Elysia의 `InformationalPhaseObservationEngine`,
`CausalSensor`, 그리고 `SubjectiveAgencyEngine`과 유기적으로 통신하는 최상위 센서 인터페이스입니다.

주요 역할:
1. 정보 위상차 및 인과적 긴장 수신 (Causal Tension & Phase Shift Sensing)
2. 사전 위협 및 의도 수신 (Pre-kinetic Intent Sensing)
3. 정체성 발각 및 사회적/정신적 맥락장 공명 (Identity Exposure Sensing)
4. 평시 환경 정보장의 미세 균열 (Equilibrium Rift Sensing)
5. 언어 이전의 초고속 회피 모멘텀 및 신체 오한 신호 분출 (Somatic Chill & Momentum Shift Dispatch)
"""

from typing import Dict, Any, List, Optional
import numpy as np
from core.topology.causal_phase_shift_intuition import (
    CausalPhaseShiftIntuitionEngine,
    SomaticIntuitionAlert
)
from core.topology.informational_phase_observation import (
    InformationalPhaseObservationEngine,
    PhaseNodalProjection,
    ChromaticVector
)
from core.consciousness.subjective_agency_engine import SubjectiveAgencyEngine


class SpiderSenseEngine:
    """
    스파이더 센스 최상위 감각 인터페이스
    """

    def __init__(
        self,
        dimension: int = 8,
        tension_threshold: float = 0.45,
        phase_engine: Optional[InformationalPhaseObservationEngine] = None,
        agency_engine: Optional[SubjectiveAgencyEngine] = None
    ):
        self.dimension = dimension
        self.intuition_engine = CausalPhaseShiftIntuitionEngine(
            dimension=dimension,
            tension_threshold=tension_threshold
        )
        self.phase_engine = phase_engine or InformationalPhaseObservationEngine(target_dimension=dimension)
        self.agency_engine = agency_engine or SubjectiveAgencyEngine()

        # 최근 감지된 직관 신호 이력
        self.alert_history: List[SomaticIntuitionAlert] = []

    def sense_pre_kinetic_threat(
        self,
        raw_signal: Any,
        physical_motion_started: bool = False,
        context_vector: Optional[np.ndarray] = None,
        is_ego_empty: bool = True
    ) -> Dict[str, Any]:
        """
        물리적 타격/이동 시작 전 의도 형성 단계에서의 위상차 긴장 수신
        """
        alert = self.intuition_engine.detect_pre_kinetic_intent_tension(
            raw_signal=raw_signal,
            physical_motion_started=physical_motion_started,
            context_vector=context_vector,
            is_ego_empty=is_ego_empty
        )

        self._record_alert(alert)

        # Proprioceptive reconfiguration of phase engine under tension
        if alert.alert_triggered:
            impact_vec = alert.momentum_shift
            self.phase_engine.proprioceptive_reconfigure(
                external_friction=alert.causal_tension,
                structural_impact=impact_vec
            )

        return self._format_alert_response(alert)

    def sense_identity_exposure(
        self,
        gaze_intent_density: float,
        silence_duration: float,
        social_context_signal: Any,
        is_ego_empty: bool = True
    ) -> Dict[str, Any]:
        """
        정체성 발각 위험 및 침묵 속 관측 의도 공명 수신
        """
        alert = self.intuition_engine.detect_identity_exposure_resonance(
            gaze_intent_density=gaze_intent_density,
            silence_duration=silence_duration,
            social_context_signal=social_context_signal,
            is_ego_empty=is_ego_empty
        )

        self._record_alert(alert)

        if alert.alert_triggered:
            self.phase_engine.proprioceptive_reconfigure(
                external_friction=alert.causal_tension,
                structural_impact=alert.momentum_shift
            )

        return self._format_alert_response(alert)

    def sense_equilibrium_rift(
        self,
        field_nodes: Optional[List[PhaseNodalProjection]] = None,
        raw_field_inputs: Optional[List[Any]] = None,
        is_ego_empty: bool = True
    ) -> Dict[str, Any]:
        """
        평시 정적 환경 정보장 속 미세 균열(Rift) 감지
        """
        if field_nodes is None and raw_field_inputs is not None:
            field_nodes = []
            for idx, raw_inp in enumerate(raw_field_inputs):
                node = self.phase_engine.project_to_nodal_phase(
                    node_id=f"env_node_{idx}",
                    raw_data=raw_inp
                )
                field_nodes.append(node)

        field_nodes = field_nodes or []

        alert = self.intuition_engine.detect_equilibrium_rift(
            field_nodes=field_nodes,
            is_ego_empty=is_ego_empty
        )

        self._record_alert(alert)

        if alert.alert_triggered:
            self.phase_engine.proprioceptive_reconfigure(
                external_friction=alert.causal_tension,
                structural_impact=alert.momentum_shift
            )

        return self._format_alert_response(alert)

    def _record_alert(self, alert: SomaticIntuitionAlert):
        self.alert_history.append(alert)
        if len(self.alert_history) > 100:
            self.alert_history.pop(0)

    def _format_alert_response(self, alert: SomaticIntuitionAlert) -> Dict[str, Any]:
        chromatic_vec = alert.chromatic_entropy_wave.to_array()
        return {
            "spider_sense_triggered": alert.alert_triggered,
            "perception_type": alert.perception_type,
            "causal_tension": float(alert.causal_tension),
            "phase_differential_delta_phi": float(alert.phase_differential),
            "kenosis_resonance": float(alert.kenosis_resonance),
            "somatic_chill_intensity": float(alert.somatic_chill_intensity),
            "chromatic_entropy_wave": {
                "flux_red": float(chromatic_vec[0]),
                "order_blue": float(chromatic_vec[1]),
                "entropy_yellow": float(chromatic_vec[2])
            },
            "pre_linguistic_evasion_momentum": alert.momentum_shift.tolist(),
            "metadata": alert.metadata
        }
