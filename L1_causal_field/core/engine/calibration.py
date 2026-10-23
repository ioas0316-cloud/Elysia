"""
Elysia Core Engine: Self-Calibration & Zero-Shot Transfer Safety Guard Engine

This module implements the Self-Calibration loop and 4-tier Safety Guard
(Semantic Dimension, Sandboxed Counterfactual Test, Axiom Invariance, Confidence Gating).
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from core.topology.grounding_ontology import GroundingAxiom, GroundingOntologyEngine
from core.topology.ast_rewriting import SelfRewritingNodeEngine


@dataclass
class CalibrationReport:
    node_id: str
    error_residual: float
    threshold: float
    recalibrated: bool
    causal_attribution: Optional[str] = None


class SelfCalibrationEngine:
    """내부 예측과 외부 반응 간의 갭을 측정하여 시스템을 자동 보정하는 중앙 엔진"""

    def __init__(
        self,
        ontology_engine: GroundingOntologyEngine,
        ast_rewriter: SelfRewritingNodeEngine,
        error_threshold: float = 0.15,
    ):
        self.ontology = ontology_engine
        self.rewriter = ast_rewriter
        self.threshold = error_threshold

    def step_and_calibrate(
        self, node_id: str, current_state: dict, real_sensor_output: dict
    ) -> Tuple[dict, CalibrationReport]:
        # 1. Forward Projection: 내부 원리망으로 결과 예측
        predicted_output = self._predict_next_state(node_id, current_state)

        # 2. Discrepancy Metric: 오차 잔차(e) 계산
        error = self._calculate_error(predicted_output, real_sensor_output)
        print(f"📊 [{node_id}] 예측 오차 잔차 측정: {error:.4f} (허용 임계치: {self.threshold})")

        # 3. Decision & Recalibration Loop
        if error > self.threshold:
            print(f"⚠️ 오차 허용치 초과! [{node_id}] 원리 구조 자가 보정 절차 진입")

            # Causal Attribution: 오류 원인 노드 및 성질 지목
            fault_cause = self._trace_causal_attribution(node_id, predicted_output, real_sensor_output)

            # AST Rewriting 또는 온톨로지 피드백 전파
            self.rewriter.rewrite_node_from_violations(node_id, [fault_cause])

            # 보정된 코드 기반 재실행
            recalibrated_output = self.rewriter.execute_node(node_id, current_state)
            report = CalibrationReport(
                node_id=node_id,
                error_residual=error,
                threshold=self.threshold,
                recalibrated=True,
                causal_attribution=fault_cause,
            )
            return recalibrated_output, report

        report = CalibrationReport(
            node_id=node_id, error_residual=error, threshold=self.threshold, recalibrated=False
        )
        return real_sensor_output, report

    def _predict_next_state(self, node_id: str, state: dict) -> dict:
        return {"val": state.get("val", 0) * 1.0}

    def _calculate_error(self, pred: dict, real: dict) -> float:
        return abs(pred.get("val", 0) - real.get("val", 0))

    def _trace_causal_attribution(self, node_id: str, pred: dict, real: dict) -> str:
        return f"ERR_DISCREPANCY_IN_{node_id}"


# ============================================================================
# 2. Zero-shot Transfer Safety Guard (4-Tier)
# ============================================================================

class ZeroShotSafetyGuard:
    """새로운 도메인 이식 시 인과 오류 및 파괴적 행동을 차단하는 4중 안전 가드"""

    def __init__(self, ontology: GroundingOntologyEngine, min_confidence_threshold: float = 0.8):
        self.ontology = ontology
        self.min_confidence_threshold = min_confidence_threshold

    def validate_transfer(
        self,
        domain_name: str,
        abstract_rule: Dict[str, Any],
        proposed_qualities: Set[str],
        proposed_bindings: Dict[str, str],
        confidence_score: float,
        sandbox_fn: Optional[Callable[[], bool]] = None,
    ) -> Tuple[bool, List[str]]:
        violations = []

        # 1단계: 시맨틱 차원 및 물리 단위 검증 (Semantic Dimensionality Check)
        if "dimension_match" in abstract_rule and not abstract_rule["dimension_match"]:
            violations.append(f"[Tier 1 Dimension Check] '{domain_name}' 도메인의 시맨틱 차원 불일치")

        # 2단계: 샌드박스 가상 격리 검증 (Sandboxed Counterfactual Test)
        if sandbox_fn and not sandbox_fn():
            violations.append(f"[Tier 2 Sandbox Test] 샌드박스 반사실 시뮬레이션 검증 실패")

        # 3단계: 절대 불변 공리 차단막 (Axiom Invariance Guard)
        is_valid_axiom, ontology_violations = self.ontology.validate_semantic_state(
            proposed_qualities, proposed_bindings
        )
        if not is_valid_axiom:
            for ov in ontology_violations:
                violations.append(f"[Tier 3 Axiom Guard] {ov}")

        # 4단계: 신뢰도 게이팅 (Confidence Gating)
        if confidence_score < self.min_confidence_threshold:
            violations.append(
                f"[Tier 4 Confidence Gating] 신뢰도 부족 ({confidence_score:.2f} < {self.min_confidence_threshold}) -> SAFE_FALLBACK 전환"
            )

        passed = len(violations) == 0
        return passed, violations
