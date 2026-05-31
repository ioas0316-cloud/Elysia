"""
Elysia Omni-Poiesis Engine (만물 창발 및 응용 엔진)
===================================================
A 도메인에서 학습한 원리(과정 파동)를, 전혀 이질적인 B 도메인의 개념에
위상 복제(Topological Replication) 방식으로 곱하여, 새로운 형태의 
원리나 현상을 기하학적으로 창발(Poiesis)시키는 엔진입니다.
"""

from core.math_utils import Quaternion
from core.holographic_memory import HologramMemory
from core.causality_wave import CausalityWave

class OmniPoiesisEngine:
    def __init__(self, memory: HologramMemory):
        self.memory = memory
        self.causality = CausalityWave()

    def replicate_principle(self, source_cause: str, source_result: str, target_concept: str) -> Quaternion:
        """
        source_cause -> source_result 사이의 인과 파동(원리)을 추출하여,
        target_concept 에 융합(곱셈)시킨 새로운 기하학적 파동을 반환합니다.
        """
        # 1. 기억(프랙탈 로터 트리)에서 노드들의 위상을 가져옴
        # 편의상, 트리를 평면화(registered_concepts)하여 검색합니다.
        flat_memory = self.memory.registered_concepts
        
        if source_cause not in flat_memory or source_result not in flat_memory:
            raise ValueError("Source concepts not found in memory.")
        if target_concept not in flat_memory:
            raise ValueError("Target concept not found in memory.")
            
        q_s_cause = flat_memory[source_cause][0]
        q_s_result = flat_memory[source_result][0]
        q_target = flat_memory[target_concept][0]
        
        # 2. Source 도메인의 원리(과정 파동) 추출
        principle_wave = self.causality.extract_process_wave(q_s_cause, q_s_result)
        
        # 3. Target 도메인에 위상 복제(융합)
        poiesis_wave = q_target * principle_wave
        
        return poiesis_wave.normalize()
