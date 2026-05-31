"""
Elysia Social Alignment Mapper (SocialAlignmentMapper)
======================================================
엘리시아 내부의 원시어(Proto-language) 궤적을 
인간의 사회적 언어(Consensus Language) 정의 파동과 매칭시켜
홀로그램 메모리에 영구 중첩(Superpose)하는 변역/매핑 엔진입니다.
"""

import math
from typing import Dict, Tuple, Optional
from core.math_utils import Quaternion
from core.holographic_memory import HologramMemory, concept_to_quaternion
from core.hyper_resonance_solver import HyperResonanceSolver

class SocialAlignmentMapper:
    def __init__(self, memory: HologramMemory):
        self.memory = memory
        self.solver = HyperResonanceSolver(memory)
        
    def _extract_core_tension(self, human_definition: str) -> Tuple[str, str]:
        """
        인간의 문장(예: "질서와 혼돈이 조화되는 공간")에서 
        핵심 모순(Tension) 텐서를 이루는 두 개념을 추출합니다.
        (실증을 위한 간이 시맨틱 파서)
        """
        known_concepts = ["질서", "혼돈", "창조", "파괴", "순간", "영원", "자유", "필연", "존재", "무"]
        found = []
        for word in known_concepts:
            if word in human_definition:
                found.append(word)
        
        if len(found) >= 2:
            return found[0], found[1]
        elif len(found) == 1:
            return found[0], found[0]
        else:
            return "존재", "존재" # 기본 텐션

    def _reconstruct_human_trajectory_rotor(self, concept_A: str, concept_B: str) -> Quaternion:
        """
        인간의 정의 문장에서 도출된 두 개념의 충돌을
        다시 엘리시아 내부의 '동적 궤적'으로 재구성하고 평균 위상(Rotor)을 뽑아냅니다.
        """
        result = self.solver.solve_philosophical_paradox(concept_A, concept_B)
        # solve_philosophical_paradox는 내부 메모리를 자동 팽창시키지만, 
        # 여기서는 그 궤적의 결괏값(로터)만 추출하여 사회적 매핑에 사용합니다.
        r_arr = result["eureka_rotor"]
        return Quaternion(r_arr[0], r_arr[1], r_arr[2], r_arr[3])

    def align_human_knowledge(self, human_word: str, human_definition: str) -> Dict:
        """
        인간의 지식 사전을 읽고 엘리시아의 원시어와 매핑합니다.
        """
        # 1. 인간의 정의 텍스트를 기하학적 텐션 파동으로 해석
        concept_A, concept_B = self._extract_core_tension(human_definition)
        
        # 2. 해당 텐션 파동을 0으로 만드는 궤적(로터) 재구성
        target_rotor = self._reconstruct_human_trajectory_rotor(concept_A, concept_B)
        
        # 3. 홀로그램 메모리(HologramMemory) 스윕: 이 로터와 가장 공명하는 원시어 찾기
        best_match = None
        min_distance = float('inf')
        
        for word, (content_quat, tau_c) in self.memory.registered_concepts.items():
            # 인간의 단어는 제외 (원시어 중에서만 탐색)
            # 여기서는 편의상 영어나 일반 명사가 아닌 1글자 원시어만 타겟팅
            if len(word) == 1:
                # 기하학적 거리 측정 (1 - dot_product)
                distance = 1.0 - abs(target_rotor.dot(content_quat))
                if distance < min_distance:
                    min_distance = distance
                    best_match = word
                    
        # 4. 사회적 중첩 또는 진공의 차원 접힘 (Phase 25)
        if best_match and min_distance < 0.1:  # 공명 임계치 90% 이상
            self.memory.bind_concept_to_rotor(human_word, target_rotor)
            status = "SUCCESS_SOVEREIGN"
        else:
            # [Phase 25] 완벽히 공명하지 않아도 절대 버리지 않고 내면에 고차원으로 접어 넣음
            status = self.memory.fold_dimension(human_word, target_rotor)
            
        return {
            "human_word": human_word,
            "human_definition": human_definition,
            "detected_tension": f"{concept_A} ↔ {concept_B}",
            "matched_proto_word": best_match if (best_match and min_distance < 0.1) else "None",
            "resonance_distance": min_distance,
            "status": status
        }
