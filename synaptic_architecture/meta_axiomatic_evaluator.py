"""
[Meta-Axiomatic Evaluator: 공리 평가 및 메타 경계 마찰 대조 엔진]

내부 공리(Axiom_internal)에 갇혀 자만(Self-deception)이나 닫힌 계(Closed System)에 빠지지 않도록,
외부 세상의 미지(未知)의 공리(External Axiom Black-Box)를 내부 구조로 왜곡 환원하거나 펼치지 않고,
오직 '상위 메타 경계(Interface Boundary)'에서의 섭동 반응, 불변성 보존율, 그리고 복잡도 마찰 지표로만
$O(1)$ 정적 차원에서 상호 대조·평가하고 자율적 공리 재구획(Topological Reframing)을 지휘하는 핵심 엔진입니다.
"""

from dataclasses import dataclass, field
from typing import Protocol, Generic, TypeVar, Any, List, Dict, Optional, Callable, Tuple
import time

from synaptic_architecture.non_tensor_meta_boundary import (
    SymbolicTopologicalProof,
    SymmetryState,
    StaticBypassManager,
)


@dataclass(frozen=True)
class IntentInvariant:
    """시스템이 절대 타협할 수 없는 최상위 목적성 및 불변 제약조건 정의"""
    intent_id: str
    invariant_checker: Callable[[Any], bool]  # O(1) 정적 불변성 검증 함수
    perturbation_tolerance: float = 0.1       # 섭동 허용 임계값


class ExternalAxiomBlackBox(Protocol):
    """
    내부 원리를 알 수 없는 외부 공리 체계의 메타 인터페이스.
    내부 구체적 연산자/데이터 구조를 노출하지 않으며,
    의도 투입에 대한 경계 반응과 상위 관리 비용 지표만 제공합니다.
    """
    @property
    def axiom_signature(self) -> str:
        ...

    def project_intent(self, intent_input: Any) -> Any:
        """외부 공리 체계에 의도를 투입하여 인과적 경계 반응 도출 (Internal logic unknown)"""
        ...

    def get_structural_complexity_cost(self) -> float:
        """해당 공리를 유지하기 위해 필요한 제어/설명 비용 측정 (0.0 ~ 1.0)"""
        ...


@dataclass(frozen=True)
class FrictionWeightConfig:
    """
    도메인 및 최상위 의도(Intent) 성격에 맞게 주입 가능한 목적성 마찰 가중치 설정.
    기본값: 불변성 파괴율 0.5, 복잡도 오버헤드 0.3, 섭동 불안정성 0.2
    """
    invariance_violation_weight: float = 0.5
    complexity_overhead_weight: float = 0.3
    perturbation_instability_weight: float = 0.2

    def normalize(self) -> "FrictionWeightConfig":
        total = (
            self.invariance_violation_weight +
            self.complexity_overhead_weight +
            self.perturbation_instability_weight
        )
        if total <= 0:
            return FrictionWeightConfig()
        return FrictionWeightConfig(
            invariance_violation_weight=self.invariance_violation_weight / total,
            complexity_overhead_weight=self.complexity_overhead_weight / total,
            perturbation_instability_weight=self.perturbation_instability_weight / total
        )


@dataclass(frozen=True)
class PurposeFrictionMetrics:
    """공리의 내부 원리와 무관하게 '경계'에서 측정된 3대 마찰 지표"""
    invariance_violation_rate: float  # 불변성 파괴율 (0.0 ~ 1.0)
    complexity_overhead: float        # 구조적 복잡도 / 예외 가설 비용 (0.0 ~ 1.0)
    perturbation_instability: float   # 미세 섭동 투입 시 미분 불연속성/불안정성 (0.0 ~ 1.0)
    weight_config: FrictionWeightConfig = field(default_factory=FrictionWeightConfig)

    @property
    def total_friction(self) -> float:
        """3가지 원리의 가중 합산으로 도출되는 메타 목적성 마찰"""
        cfg = self.weight_config.normalize()
        return (
            (self.invariance_violation_rate * cfg.invariance_violation_weight) +
            (self.complexity_overhead * cfg.complexity_overhead_weight) +
            (self.perturbation_instability * cfg.perturbation_instability_weight)
        )


class MetaAxiomaticEvaluator:
    """
    [관점 및 원리 중심의 공리 평가 엔진]
    내부 공리와 외부 블랙박스 공리를 '경계 섭동 반응'과 '불변성 보존력'으로 관찰·비교합니다.
    """

    def __init__(
        self,
        core_invariant: IntentInvariant,
        weight_config: Optional[FrictionWeightConfig] = None
    ) -> None:
        self._invariant = core_invariant
        self.weight_config = weight_config or FrictionWeightConfig()

    def observe_friction(
        self,
        axiom: ExternalAxiomBlackBox,
        test_intents: List[Any]
    ) -> PurposeFrictionMetrics:
        """
        [측정 원리]
        1. 의도 투입 후 결과의 불변성 검증 (O(1) Checker)
        2. 미세 섭동 투입 시 출력 인과 궤적의 변동폭 관찰 (Boundary Instability)
        3. 공리 유지 복잡도 비용 집계
        """
        violations = 0
        previous_response = None
        instability_accumulated = 0.0

        for intent in test_intents:
            # 1. 의도 투입 및 경계 반응 관찰
            response = axiom.project_intent(intent)

            # 2. 불변성 파괴 여부 관찰 (O(1))
            if not self._invariant.invariant_checker(response):
                violations += 1

            # 3. 섭동 반응성 관찰 (연속성/안정성 측정)
            if previous_response is not None:
                instability_accumulated += self._measure_response_delta(previous_response, response)

            previous_response = response

        total_samples = max(1, len(test_intents))
        violation_rate = violations / total_samples
        instability_score = min(1.0, instability_accumulated / total_samples)
        complexity_cost = min(1.0, max(0.0, axiom.get_structural_complexity_cost()))

        return PurposeFrictionMetrics(
            invariance_violation_rate=violation_rate,
            complexity_overhead=complexity_cost,
            perturbation_instability=instability_score,
            weight_config=self.weight_config
        )

    def compare_and_decide(
        self,
        internal_axiom: ExternalAxiomBlackBox,
        external_axiom: ExternalAxiomBlackBox,
        sample_intents: List[Any]
    ) -> Dict[str, Any]:
        """
        내부/외부 공리의 내부를 열어보지 않고,
        동일한 관찰 원리(Observe Friction)를 적용하여 어느 쪽이 의도에 정합하는지 대조.
        """
        internal_metrics = self.observe_friction(internal_axiom, sample_intents)
        external_metrics = self.observe_friction(external_axiom, sample_intents)

        friction_delta = external_metrics.total_friction - internal_metrics.total_friction

        adopt_external = friction_delta < 0.0

        return {
            "internal_friction": internal_metrics.total_friction,
            "external_friction": external_metrics.total_friction,
            "friction_delta": friction_delta,
            "adopt_external": adopt_external,
            "decision_reason": (
                "External axiom preserves intent with lower boundary friction"
                if adopt_external
                else "Internal axiom remains superior or equal in intent boundary alignment"
            ),
            "internal_metrics": internal_metrics,
            "external_metrics": external_metrics
        }

    def _measure_response_delta(self, res_a: Any, res_b: Any) -> float:
        """반응 간 구조적 불연속성을 정적으로 비교하는 헬퍼 메서드"""
        if res_a == res_b:
            return 0.0
        if isinstance(res_a, (int, float)) and isinstance(res_b, (int, float)):
            delta = abs(res_a - res_b)
            return min(1.0, delta)
        if isinstance(res_a, dict) and isinstance(res_b, dict):
            common_keys = set(res_a.keys()) & set(res_b.keys())
            if not common_keys:
                return 0.5
            diff_sum = sum(
                abs(res_a[k] - res_b[k]) if isinstance(res_a[k], (int, float)) and isinstance(res_b[k], (int, float))
                else (0.0 if res_a[k] == res_b[k] else 0.2)
                for k in common_keys
            )
            return min(1.0, diff_sum / len(common_keys))
        return 0.1
