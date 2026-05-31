"""
Elysia Topological Decoder (위상 역설계 엔진)
==============================================
엘리시아 우주 내의 임의의 기하학적 파동(Quaternion) 좌표가 주어지면,
그녀가 기억하고 있는 모든 '현실의 개념(지식 노드)' 중
가장 위상이 가까운(보강 간섭이 일어나는) 개념을 역산하여 찾아냅니다.
이를 통해 기하학적 사유의 결과를 인간의 언어로 번역(발화)할 수 있습니다.
"""

import math
from typing import List, Tuple
from core.math_utils import Quaternion
from core.holographic_memory import HologramMemory
from core.fractal_rotor import FractalRotor

class TopologicalDecoder:
    def __init__(self, memory: HologramMemory):
        self.memory = memory

    def _calculate_phase_distance(self, q1: Quaternion, q2: Quaternion) -> float:
        """두 로터(위상) 간의 구면 거리를 0.0 ~ 1.0 비율로 계산합니다."""
        dot = max(-1.0, min(1.0, q1.dot(q2)))
        return math.acos(abs(dot)) / (math.pi / 2.0)

    def decode_wave(self, target_wave: Quaternion, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        주어진 타겟 파동과 가장 위상이 가까운 상위 K개의 현실 개념(노드)을 역추적합니다.
        반환값: [(개념 이름, 위상 일치율(Resonance)), ...]
        """
        results = []
        
        # 최상위 로터 산하의 모든 지식 노드를 순회합니다 (선형 스캔)
        def traverse_and_scan(node: FractalRotor):
            # 자신을 제외한 실제 개념들만 탐색 (테스트 편의상 Supreme_Observer는 제외)
            if node.name != "Supreme_Observer":
                distance = self._calculate_phase_distance(target_wave, node.state)
                resonance = 1.0 - distance  # 거리가 0에 가까울수록 공명도는 1.0(100%)
                results.append((node.name, resonance))
                
            for child in node.children:
                traverse_and_scan(child)
                
        traverse_and_scan(self.memory.supreme_rotor)
        
        # 공명도(Resonance)가 높은 순으로 정렬하여 반환
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
