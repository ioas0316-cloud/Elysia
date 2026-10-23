"""
Elysia Core Architecture: VQ Sensory Projector

This module projects continuous sensory vectors (R^d) into discrete symbolic qualities
via vector quantization (VQ) prototype metric comparison and dynamic activation thresholds.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass
class SymbolPrototype:
    symbol_name: str
    codebook_vector: List[float]
    threshold: float  # 동적 발화 임계값 (tau)


class VQSensoryProjector:
    """연속 감각 벡터를 프로토타입 거리 기반 이산 기호로 양자화(VQ)하는 프로젝터"""

    def __init__(self):
        self.prototypes: Dict[str, SymbolPrototype] = {}

    def register_prototype(self, symbol_name: str, vector: List[float], initial_threshold: float):
        self.prototypes[symbol_name] = SymbolPrototype(symbol_name, vector, initial_threshold)

    def _euclidean_distance(self, v1: List[float], v2: List[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(v1, v2)))

    def project_to_qualities(self, sensory_vector: List[float]) -> Tuple[Set[str], Dict[str, float]]:
        """감각 벡터 -> 이산 Qualities 발화 및 거리 메트릭 반환"""
        active_qualities = set()
        distances = {}

        for symbol_name, proto in self.prototypes.items():
            dist = self._euclidean_distance(sensory_vector, proto.codebook_vector)
            distances[symbol_name] = dist

            # 프로토타입과의 거리가 임계값(tau) 이하일 때 기호 발화
            if dist <= proto.threshold:
                active_qualities.add(symbol_name)

        return active_qualities, distances

    def adapt_threshold(self, symbol_name: str, contraction_factor: float = 0.7):
        """Top-down 피드백 수신 시 특정 기호의 발화 임계값(tau) 수축"""
        if symbol_name in self.prototypes:
            proto = self.prototypes[symbol_name]
            old_tau = proto.threshold
            proto.threshold *= contraction_factor
            print(f"  └─ 📉 [Dual Plasticity] '{symbol_name}' 발화 임계값 수축: {old_tau:.3f} -> {proto.threshold:.3f}")
