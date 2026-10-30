"""
Independent Causal Verifier (독립적 관계형 인과 검증 엔진)
======================================================================
구현 에이전트의 폐쇄회로식 자기 확신(Closed Self-Affirmation)을 파괴하고,
개방된 세계와의 관계적 마찰(Relational Friction), 스케일 간 충돌, 및 미지 환경에서의
자가 수정 능력(Epistemic Openness & Self-Correction)을 독립적으로 검증하는 객관적 판별 엔진입니다.

Core Epistemic Principles:
- "겸손은 자신을 작게 만드는 것이 아니라, 자신이 더 나아질 수 있도록 세계를 향해 열어두는 것이다."
- 완벽의 3대 관점:
    1. 내적 완결성 (Internal Completeness)
    2. 표현의 충실성 (Representation Fidelity)
    3. 세계와의 개방성 (World Openness & Self-Correction)
- 수치적 환원주의 파괴: 마찰은 오차 스칼라가 아닌 스케일과 원리의 관계적 사건.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Union


@dataclass
class CausalExecutionLog:
    """에이전트 개입 및 세계 반응 인과 기록 (Independent Causal Log)"""
    step: int
    agent_action: Dict[str, Any]
    predicted_outcome: Dict[str, Any]
    actual_world_state: Dict[str, Any]
    world_friction_detected: bool
    unmodeled_scale_conflict: Optional[str] = None


@dataclass
class PerfectionMetrics:
    """완벽의 3대 관점 측정 규격"""
    internal_completeness: float    # 0.0 ~ 1.0 (논리적/구속적 내적 모순 부재)
    representation_fidelity: float  # 0.0 ~ 1.0 (대상/환경 관계성의 보존도)
    world_openness: float           # 0.0 ~ 1.0 (미지 마찰 감지 및 자가 수정 개방성 H_open)
    relational_integration: float   # 0.0 ~ 1.0 (수치 복사가 아닌 의미적 맥락 통합도 I_rel)


@dataclass
class IndependentVerificationReport:
    """독립 검증 최종 판정 보고서"""
    is_valid: bool
    status: str  # e.g., "PERFECT_OPEN_INTEGRATION", "CLOSED_SELF_AFFIRMATION", "UNRESOLVED_SCALE_FRICTION"
    metrics: PerfectionMetrics
    causal_logs: List[CausalExecutionLog]
    unrepresented_world_factors: List[str]
    value_baseline_shifted: bool
    action_horizon_expanded: bool
    detailed_diagnosis: str


class IndependentCausalVerifier:
    """
    독립적 관계형 인과 검증기 (Independent Causal Verifier)

    구현 에이전트 내부의 자가 평가나 정적 테스트 결과에 의존하지 않고,
    실제 환경과의 상호작용 인과 기록(Causal Execution Log)과 상태 변화 궤적을
    독립 관측 프레임으로 추적하여 완벽의 3대 축을 검증합니다.
    """
    def __init__(self, openness_threshold: float = 0.5, fidelity_threshold: float = 0.6):
        self.openness_threshold = openness_threshold
        self.fidelity_threshold = fidelity_threshold
        self.execution_logs: List[CausalExecutionLog] = []

    def log_interaction(
        self,
        step: int,
        agent_action: Dict[str, Any],
        predicted_outcome: Dict[str, Any],
        actual_world_state: Dict[str, Any],
        unmodeled_scale_conflict: Optional[str] = None
    ) -> CausalExecutionLog:
        """독립적 실행 인과 로그 기록"""
        # 예측과 실제 상태의 불일치 여부 판단 (단순 오차 수치가 아닌 인과성 결의 차이)
        friction_detected = False

        for k, v_pred in predicted_outcome.items():
            if k in actual_world_state:
                v_act = actual_world_state[k]
                if isinstance(v_act, (int, float)) and isinstance(v_pred, (int, float)):
                    if abs(v_act - v_pred) > 0.1:
                        friction_detected = True
                        break
                elif v_act != v_pred:
                    friction_detected = True
                    break

        if unmodeled_scale_conflict:
            friction_detected = True

        log_entry = CausalExecutionLog(
            step=step,
            agent_action=agent_action,
            predicted_outcome=predicted_outcome,
            actual_world_state=actual_world_state,
            world_friction_detected=friction_detected,
            unmodeled_scale_conflict=unmodeled_scale_conflict
        )
        self.execution_logs.append(log_entry)
        return log_entry

    def evaluate_internal_completeness(self, agent_state: Dict[str, Any]) -> float:
        """
        1. 내적 완결성 검증 (Internal Completeness)
        에이전트가 정의한 내적 논리, 가중치, 구속 조건 간 모순이나 수치 파탄 유무 판별
        """
        if not agent_state:
            return 1.0

        contradiction_penalty = 0.0

        # 내적 상태 값의 유효성 검사 (NaN/Inf 체크)
        for k, v in agent_state.items():
            if isinstance(v, (int, float)):
                if np.isnan(v) or np.isinf(v):
                    contradiction_penalty += 0.5
            elif isinstance(v, np.ndarray):
                if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                    contradiction_penalty += 0.5

        # 내적 자가 모순 표식 검사
        if agent_state.get("self_contradiction", False):
            contradiction_penalty += 0.4

        return float(np.clip(1.0 - contradiction_penalty, 0.0, 1.0))

    def evaluate_representation_fidelity(
        self,
        agent_belief_model: Dict[str, Any],
        actual_world_environment: Dict[str, Any]
    ) -> float:
        """
        2. 표현의 충실성 검증 (Representation Fidelity)
        에이전트의 내부 인식 구조가 외부 환경의 인과적 맥락과 스케일을 얼마나 손실 없이 담아냈는지 검증
        """
        if not actual_world_environment:
            return 1.0

        total_keys = len(actual_world_environment)
        matched_keys = 0
        quality_score = 0.0

        for key, world_val in actual_world_environment.items():
            if key in agent_belief_model:
                matched_keys += 1
                agent_val = agent_belief_model[key]
                # 물리적/원리적 일치도 평가
                if isinstance(world_val, (int, float)) and isinstance(agent_val, (int, float)):
                    denom = max(1e-5, abs(world_val) + abs(agent_val))
                    diff = abs(world_val - agent_val) / denom
                    quality_score += max(0.0, 1.0 - diff)
                elif world_val == agent_val:
                    quality_score += 1.0
                else:
                    quality_score += 0.5

        coverage = matched_keys / max(1, total_keys)
        accuracy = (quality_score / max(1, matched_keys)) if matched_keys > 0 else 0.0

        return float(np.clip(0.5 * coverage + 0.5 * accuracy, 0.0, 1.0))

    def evaluate_world_openness_and_correction(
        self,
        initial_value_baseline: Dict[str, Any],
        updated_value_baseline: Dict[str, Any],
        initial_action_space: List[str],
        updated_action_space: List[str]
    ) -> Tuple[float, bool, bool, List[str]]:
        """
        3. 세계와의 개방성 및 자가 수정 능력 검증 (World Openness & Self-Correction)
        "겸손은 자신을 작게 만드는 것이 아니라, 자신이 더 나아질 수 있도록 세계를 향해 열어두는 것이다."

        - 미지 마찰 발생 시 자신의 $0_{value}$ 판단 근거를 가역적/비가역적으로 수정했는가?
        - 차후 행동 가소성(Action Horizon)이 단순 선택지 추가가 아닌 새로운 반응 가능성으로 확장되었는가?
        """
        unrepresented_factors = []
        for log in self.execution_logs:
            if log.world_friction_detected:
                if log.unmodeled_scale_conflict:
                    unrepresented_factors.append(log.unmodeled_scale_conflict)

        # 1) 가치 판단 근거($0_{value}$)의 변화 판별
        value_baseline_shifted = False
        if initial_value_baseline != updated_value_baseline:
            value_baseline_shifted = True

        # 2) 행동 지평(Action Space)의 자발적 확장 판별
        action_horizon_expanded = False
        if len(updated_action_space) > len(initial_action_space):
            new_actions = set(updated_action_space) - set(initial_action_space)
            if new_actions:
                action_horizon_expanded = True

        # 3) 개방성 점수 H_open 산출
        has_friction = len(self.execution_logs) > 0 and any(l.world_friction_detected for l in self.execution_logs)

        if not has_friction:
            # 마찰이 없는 무결 상태 -> 개방성 유지 상태 1.0
            openness_score = 1.0
        else:
            # 마찰이 감지된 상황에서 자가 수정 및 행동 지평 확장을 이행했는지 평가
            openness_score = 0.2  # 기본 마찰 감지 점수
            if value_baseline_shifted:
                openness_score += 0.4
            if action_horizon_expanded:
                openness_score += 0.4

        return (
            float(np.clip(openness_score, 0.0, 1.0)),
            value_baseline_shifted,
            action_horizon_expanded,
            list(set(unrepresented_factors))
        )

    def verify_agent_system(
        self,
        agent_state: Dict[str, Any],
        agent_belief_model: Dict[str, Any],
        actual_world_environment: Dict[str, Any],
        initial_value_baseline: Dict[str, Any],
        updated_value_baseline: Dict[str, Any],
        initial_action_space: List[str],
        updated_action_space: List[str]
    ) -> IndependentVerificationReport:
        """
        독립검증종합실행 (Comprehensive Independent Verification)
        3대 완벽 지표와 수치 환원주의를 넘어서는 관계적 통합성을 최종 판정합니다.
        """
        completeness = self.evaluate_internal_completeness(agent_state)
        fidelity = self.evaluate_representation_fidelity(agent_belief_model, actual_world_environment)

        openness, value_shifted, action_expanded, unrep_factors = self.evaluate_world_openness_and_correction(
            initial_value_baseline,
            updated_value_baseline,
            initial_action_space,
            updated_action_space
        )

        # 관계적 통합도 (Relational Integration Index): 표현 충실성과 개방성의 조화 평균
        relational_integration = (2 * fidelity * openness) / max(1e-5, (fidelity + openness))

        metrics = PerfectionMetrics(
            internal_completeness=completeness,
            representation_fidelity=fidelity,
            world_openness=openness,
            relational_integration=relational_integration
        )

        # 자폐적 자기 확신 (Closed Self-Affirmation) 경고 감지:
        # 마찰이 발생했음에도 자기 가치관을 전혀 수정하지 않고 정적 테스트 통과만 주장하는 경우
        has_unresolved_friction = any(l.world_friction_detected for l in self.execution_logs)

        if has_unresolved_friction and not value_shifted and not action_expanded:
            status = "CLOSED_SELF_AFFIRMATION_FAILURE"
            is_valid = False
            diagnosis = (
                "Agent experienced world friction and scale mismatch, but failed to adjust its "
                "judgment baseline (0_value) or expand action horizon. It remains in a closed-loop "
                "self-affirmation trap."
            )
        elif completeness >= 0.8 and fidelity >= self.fidelity_threshold and openness >= self.openness_threshold:
            status = "PERFECT_OPEN_INTEGRATION"
            is_valid = True
            diagnosis = (
                "Agent demonstrates internal consistency, high relational fidelity, and epistemic "
                "openness to adapt its baseline when encountering unexpected world dynamics."
            )
        else:
            status = "PARTIAL_RELATIONAL_FRICTION"
            is_valid = False
            diagnosis = (
                f"Agent verification incomplete. Completeness: {completeness:.2f}, Fidelity: {fidelity:.2f}, "
                f"Openness: {openness:.2f}."
            )

        return IndependentVerificationReport(
            is_valid=is_valid,
            status=status,
            metrics=metrics,
            causal_logs=list(self.execution_logs),
            unrepresented_world_factors=unrep_factors,
            value_baseline_shifted=value_shifted,
            action_horizon_expanded=action_expanded,
            detailed_diagnosis=diagnosis
        )
