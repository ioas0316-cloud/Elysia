"""
Elysia Core Architecture: OOD Latent Buffer & Axiom Emergence Engine

This module handles Out-of-Distribution (OOD) unknown sensory buffering,
3-tier noise filtering (Temporal/Spatial Consistency, Cross-Modal Causality,
Residual Entropy/Rank/Compressibility), density clustering for Symbol Genesis,
and automatic Axiom Induction into the Grounding Ontology.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================================
# 1. OOD Latent Buffer & Noise Filters
# ============================================================================

@dataclass
class OODSample:
    sensory_vector: List[float]
    timestamp_ms: float
    modalities: Dict[str, List[float]] = field(default_factory=dict)
    context_tags: Set[str] = field(default_factory=set)


class OODLatentBuffer:
    """[1단계] 기존 상식으로 해석되지 않는 '생소한 감각 데이터'를 모아두는 임시 기억 장소"""

    def __init__(self, capacity: int = 100):
        self.buffer: List[OODSample] = []
        self.capacity = capacity

    def add_unknown_signal(self, sample: OODSample):
        """기존 기호로 설명되지 않는 신호를 버퍼에 쌓음"""
        self.buffer.append(sample)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)


class ConsistencyFilter:
    """시간/공간 일관성 필터: 신호가 재현 가능한가 판별"""

    def __init__(self, min_occurrences: int = 3, max_time_window_ms: float = 5000.0):
        self.min_occurrences = min_occurrences
        self.max_time_window_ms = max_time_window_ms

    def evaluate(self, samples: List[OODSample]) -> bool:
        if len(samples) < self.min_occurrences:
            return False
        time_span = samples[-1].timestamp_ms - samples[0].timestamp_ms
        return time_span <= self.max_time_window_ms


class CrossModalCausalFilter:
    """교차 감각 인과성 필터: 행동(Action) 수행 시 다중 센서 동시 반응 파악"""

    def evaluate(self, sample: OODSample) -> bool:
        # 2개 이상의 modality (e.g. visual + motor)가 유의미하게 튀는가
        active_modalities = 0
        for mod, vec in sample.modalities.items():
            magnitude = math.sqrt(sum(x ** 2 for x in vec))
            if magnitude > 0.1:
                active_modalities += 1
        return active_modalities >= 2 or len(sample.modalities) == 0  # fall back if modal not provided


class ResidualCompressibilityFilter:
    """
    잔차 구조화 필터: 잔차 데이터 오차에 엔트로피/고유차원/압축 질서가 존재하는가 판별
    - Entropy low (< threshold)
    - Low-rank Intrinsic Dimension (PCA eigenvalues concentration)
    - High compression ratio
    """

    def evaluate(self, vectors: List[List[float]]) -> Dict[str, Any]:
        if not vectors or len(vectors) < 2:
            return {"is_valid_pattern": False, "entropy": 1.0, "rank": 99, "compressibility": 0.0}

        dim = len(vectors[0])
        n = len(vectors)

        # 1. 정보 엔트로피 계산 (각 차원의 분산 기준)
        means = [sum(vec[i] for vec in vectors) / n for i in range(dim)]
        vars_ = [sum((vec[i] - means[i]) ** 2 for vec in vectors) / n for i in range(dim)]
        total_var = sum(vars_) + 1e-9

        # normalize variances to compute entropy
        probs = [v / total_var for v in vars_]
        entropy = -sum(p * math.log2(p + 1e-9) for p in probs if p > 0)

        # 2. 고유 차원 (Intrinsic Dimension - 주성분 분산 비율)
        sorted_vars = sorted(vars_, reverse=True)
        cum_ratio = 0.0
        rank = 0
        for v in sorted_vars:
            cum_ratio += v / total_var
            rank += 1
            if cum_ratio >= 0.8:  # 80% 분산 설명 축 수
                break

        # 3. 오토인코더 / PCA 압축 복원율 추정 (Top rank축으로 복원 시 오차)
        compressibility = max(0.0, 1.0 - (rank / float(dim)))

        is_valid_pattern = (rank <= max(1, dim // 2)) or (compressibility >= 0.3)

        return {
            "is_valid_pattern": is_valid_pattern,
            "entropy": entropy,
            "rank": rank,
            "compressibility": compressibility,
        }


# ============================================================================
# 2. Axiom Inducer Engine (Emergence)
# ============================================================================

class AxiomInducer:
    """[2~5단계] 모인 생소한 감각을 3중 필터링 후 신규 기호 생성 및 공리 승격 메커니즘"""

    def __init__(
        self,
        buffer: OODLatentBuffer,
        consistency_filter: Optional[ConsistencyFilter] = None,
        cross_modal_filter: Optional[CrossModalCausalFilter] = None,
        residual_filter: Optional[ResidualCompressibilityFilter] = None,
    ):
        self.buffer = buffer
        self.consistency_filter = consistency_filter or ConsistencyFilter()
        self.cross_modal_filter = cross_modal_filter or CrossModalCausalFilter()
        self.residual_filter = residual_filter or ResidualCompressibilityFilter()

    def discover_new_concept(self) -> Optional[Dict[str, Any]]:
        """생소함 데이터 수집 -> 3중 필터 -> 밀도 중심점 개념화 -> 공리 추론"""
        if len(self.buffer.buffer) < 5:
            return None

        samples = list(self.buffer.buffer)
        vectors = [s.sensory_vector for s in samples]

        # Filter 1: Temporal / Spatial Consistency
        if not self.consistency_filter.evaluate(samples):
            print("  └─ ❌ [Noise Filter] 시간/공간 일관성 부족으로 노이즈 판정")
            return None

        # Filter 2: Cross-modal Causal Check
        valid_cross_modal = sum(1 for s in samples if self.cross_modal_filter.evaluate(s))
        if valid_cross_modal < len(samples) * 0.5:
            print("  └─ ❌ [Noise Filter] 교차 감각 인과성 부족으로 노이즈 판정")
            return None

        # Filter 3: Residual Compressibility / Low-Rank Structure Check
        residual_metrics = self.residual_filter.evaluate(vectors)
        if not residual_metrics["is_valid_pattern"]:
            print(f"  └─ ❌ [Noise Filter] 잔차 무작위성 높음 (Rank: {residual_metrics['rank']}, Compressibility: {residual_metrics['compressibility']:.2f})")
            return None

        # Step 1: Symbol Genesis (Centroid computation)
        n = len(vectors)
        dim = len(vectors[0])
        centroid = [sum(vec[i] for vec in vectors) / n for i in range(dim)]

        symbol_hash = abs(hash(tuple(round(x, 3) for x in centroid))) % 10000
        new_symbol_name = f"EMERGENT_CONCEPT_{symbol_hash}"

        # Step 2 & 3: Causal Mining & Proposition
        new_axiom_data = {
            "symbol": new_symbol_name,
            "prototype_vector": centroid,
            "initial_tau": 0.5,
            "inferred_rule": f"IF {new_symbol_name} DETECTED -> TRIGGER CAUTION_STATE",
            "metrics": residual_metrics,
        }

        self.buffer.buffer.clear()
        print(f"✨ [New Axiom Emerged] 신규 기호 '{new_symbol_name}' 및 인과 공리가 발현되었습니다!")
        print(f"   ├─ Residual Rank: {residual_metrics['rank']} / {dim}")
        print(f"   └─ Compressibility: {residual_metrics['compressibility']:.2f}")

        return new_axiom_data
