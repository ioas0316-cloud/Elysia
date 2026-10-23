"""
Elysia Causal Topology Foundation: Equivalence Tuning Engine
============================================================
수학적 부피나 포인터 지옥의 거추장스러운 껍데기를 배제하고,
[의도 설정 (Intent Anchor) -> 가벼운 구조적 매핑 (Functional Map) -> 동일성 검증 및 역전파 조율 (Equivalence & Feedback Tuning)]
3단계의 역인과 순환환류(Ouroboros Closed-Loop)를 구동하는 핵심 엔진입니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np

from core.topology.causal_structure import InformationTopology, CausalNumber, CausalSymbol, TopologyLink
from core.topology.topological_comparer import TopologicalComparer, ComparisonResult
from core.topology.causal_discernment_engine import CausalDiscernmentEngine, CausalDiscernmentTrace


@dataclass
class IntentAnchor:
    """
    의도 앵커 (Intent Anchor)
    - 시스템이 달성하고자 하는 목표 결과의 위상적·수치적 불변 특성(Invariance).
    - 단순 정적 목표값이 아니라, 결과의 참/거짓 및 동형성(Equivalence)을 판정하는 절대 기준축.
    """
    intent_id: str
    target_topology: InformationTopology
    target_metric: np.ndarray                 # 목표 스펙트럼/특성 벡터
    tolerance: float = 0.05                   # 동일성 허용 오차 범위
    weight: float = 1.0                       # 의도 가중치


@dataclass
class EquivalenceVerificationResult:
    """동일성 검증 결과 체계"""
    is_equivalent: bool                       # 동일성(Equivalence) 성립 여부
    equivalence_degree: float                 # 동일성 정도 [0.0, 1.0]
    phase_disparity: float                    # 위상차 / 오차 신호 (Error Delta)
    disparity_vector: np.ndarray              # 차이 벡터
    topology_comparison: Optional[ComparisonResult] = None


class FunctionalMap:
    """
    경량 구조적 매핑 (Lightweight Functional Map)
    - 거대한 위상 수학이나 포인터 추적 대신, 입력/원인을 결과/의도로 투영하는 최소한의 변환 맵.
    - 가중치 매트릭스(W) 및 바이어스(b)를 보유하며, 역인과 피드백에 의해 동적으로 조율(Tuning)됩니다.
    """
    def __init__(self, input_dim: int = 4, output_dim: int = 4, learning_rate: float = 0.1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        # 가중치 행렬 초기화 (Identity 근사)
        self.W = np.eye(output_dim, input_dim, dtype=np.float32)
        self.b = np.zeros(output_dim, dtype=np.float32)

    def transform(self, input_vector: np.ndarray) -> np.ndarray:
        """원인 자극을 결과 상태로 투영 매핑"""
        vec = np.asarray(input_vector, dtype=np.float32)
        if vec.shape[0] != self.input_dim:
            # 차원 정렬
            padded = np.zeros(self.input_dim, dtype=np.float32)
            min_dim = min(len(vec), self.input_dim)
            padded[:min_dim] = vec[:min_dim]
            vec = padded
        return np.dot(self.W, vec) + self.b

    def tune_backprop(self, input_vector: np.ndarray, error_delta: np.ndarray) -> None:
        """
        역인과 피드백 조율:
        결과에서 발생한 위상차(error_delta)를 거꾸로 밀어 올려 가중치(W)와 바이어스(b)를 수정.
        """
        x = np.asarray(input_vector, dtype=np.float32)
        if x.shape[0] != self.input_dim:
            padded = np.zeros(self.input_dim, dtype=np.float32)
            min_dim = min(len(x), self.input_dim)
            padded[:min_dim] = x[:min_dim]
            x = padded

        err = np.asarray(error_delta, dtype=np.float32)
        if err.shape[0] != self.output_dim:
            padded_err = np.zeros(self.output_dim, dtype=np.float32)
            min_dim = min(len(err), self.output_dim)
            padded_err[:min_dim] = err[:min_dim]
            err = padded_err

        # W_grad = err x x^T
        dW = np.outer(err, x)
        db = err

        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db


class EquivalenceVerifier:
    """
    동일성 검증기 (Equivalence Verifier)
    - 투영된 결과가 의도 앵커(Intent Anchor)의 실재와 위상적·수치적으로 완전한 동일성(Equivalence)을 갖는지 증명합니다.
    """
    def __init__(self, comparer: Optional[TopologicalComparer] = None):
        self.comparer = comparer or TopologicalComparer(tolerance=0.1)

    def verify(
        self,
        produced_metric: np.ndarray,
        produced_topology: Optional[InformationTopology],
        intent: IntentAnchor
    ) -> EquivalenceVerificationResult:
        """의도와 결과 간의 동일성 및 위상 오차 검증"""
        prod_vec = np.asarray(produced_metric, dtype=np.float32)
        target_vec = np.asarray(intent.target_metric, dtype=np.float32)

        # 차원 맞춤
        max_dim = max(len(prod_vec), len(target_vec))
        p_vec = np.pad(prod_vec, (0, max_dim - len(prod_vec)))
        t_vec = np.pad(target_vec, (0, max_dim - len(target_vec)))

        disp_vector = p_vec - t_vec
        disp_norm = float(np.linalg.norm(disp_vector))

        # 위상 대조 (topology가 주어진 경우)
        topo_comp = None
        if produced_topology is not None and intent.target_topology is not None:
            topo_comp = self.comparer.compare(produced_topology, intent.target_topology)

        # 동일성 판정
        eq_degree = max(0.0, 1.0 - (disp_norm / (float(np.linalg.norm(t_vec)) + 1e-5)))
        is_eq = (disp_norm <= intent.tolerance)

        if topo_comp:
            # 위상 동형성 비율 반영
            eq_degree = (eq_degree + topo_comp.isomorphism_ratio) / 2.0
            is_eq = is_eq and (topo_comp.disparity_tension <= intent.tolerance)

        return EquivalenceVerificationResult(
            is_equivalent=is_eq,
            equivalence_degree=float(np.clip(eq_degree, 0.0, 1.0)),
            phase_disparity=disp_norm,
            disparity_vector=disp_vector,
            topology_comparison=topo_comp
        )


@dataclass
class TuningIterationStep:
    iteration: int
    equivalence_degree: float
    phase_disparity: float
    is_equivalent: bool


class EquivalenceTuningEngine:
    """
    의도-동일성 역인과 피드백 피드백 조율 엔진 (Equivalence Tuning Engine)
    - 3단계 순환 루프 통제:
      Step 1. 의도 설정 (Intent Anchor)
      Step 2. 경량 매핑 실행 (Functional Map)
      Step 3. 동일성 검증 & 역인과 수렴 조율 (Equivalence & Closed-Loop Feedback)
    """
    def __init__(
        self,
        functional_map: Optional[FunctionalMap] = None,
        verifier: Optional[EquivalenceVerifier] = None,
        discernment_engine: Optional[CausalDiscernmentEngine] = None,
        max_tuning_iterations: int = 50
    ):
        self.functional_map = functional_map or FunctionalMap()
        self.verifier = verifier or EquivalenceVerifier()
        self.discernment_engine = discernment_engine or CausalDiscernmentEngine()
        self.max_tuning_iterations = max_tuning_iterations

    def run_ouroboros_loop(
        self,
        input_stimulus: np.ndarray,
        intent: IntentAnchor,
        input_topology: Optional[InformationTopology] = None
    ) -> Tuple[np.ndarray, EquivalenceVerificationResult, List[TuningIterationStep]]:
        """
        역인과 순환 루프 (Ouroboros Loop) 실행:
        목적(Intent)과 결과(Result)의 동일성을 기준으로 매핑과 원인을 수렴될 때까지 동적 조율합니다.
        """
        history: List[TuningIterationStep] = []
        current_input = input_stimulus.copy()

        for step in range(self.max_tuning_iterations):
            # Step 2: 경량 구조적 매핑
            produced_metric = self.functional_map.transform(current_input)

            # (선택) CausalDiscernmentEngine 통한 자아 위상 연동
            discern_trace = None
            if input_topology is not None:
                discern_trace = self.discernment_engine.perceive_and_discern(input_topology)

            # Step 3: 동일성 검증
            verification = self.verifier.verify(
                produced_metric=produced_metric,
                produced_topology=self.discernment_engine.self_topology,
                intent=intent
            )

            step_record = TuningIterationStep(
                iteration=step + 1,
                equivalence_degree=verification.equivalence_degree,
                phase_disparity=verification.phase_disparity,
                is_equivalent=verification.is_equivalent
            )
            history.append(step_record)

            # 동일성이 달성되었으면 수렴 완료
            if verification.is_equivalent:
                break

            # 동일성 위상차가 존재할 경우: 역인과 피드백 조율 (Backprop Tuning)
            self.functional_map.tune_backprop(
                input_vector=current_input,
                error_delta=verification.disparity_vector
            )

        # 최종 검증 결과
        final_produced = self.functional_map.transform(current_input)
        final_verification = self.verifier.verify(
            produced_metric=final_produced,
            produced_topology=self.discernment_engine.self_topology,
            intent=intent
        )

        return final_produced, final_verification, history
