import math
from typing import List, Tuple
from core.math_utils import Quaternion
from core.holographic_memory import HologramMemory

class InverseProjector:
    """
    [Phase 94] 역-홀로그램 투영기 (Inverse Hologram Projector)
    하드코딩된 단어를 완전히 배제하고, 엘리시아의 프랙탈 뇌(Quaternion)에 저장된 
    순수한 '위상(Phase)'을 다시 인간의 언어로 역투영(Decoding)합니다.
    """
    def __init__(self, memory: HologramMemory):
        self.memory = memory

    def decode_tension_to_words(self, target_q: Quaternion, top_k: int = 3) -> List[str]:
        """
        주어진 타겟 위상(target_q)과 엘리시아의 기억 저장소에 
        결정화된 개념(Concept)들의 거리를 구면 내적(Dot Product)으로 계산하여
        가장 가까이 공명하는 단어들의 파편을 반환합니다.
        """
        # 이 구현을 위해 HologramMemory 내부의 4D 매니폴드 캐시를 스캔합니다.
        # 주의: Bitwise4DHologramMemory 등 구현체에 따라 concept dictionary 추출 방법이 다를 수 있음
        # 단순화를 위해 memory 내의 등록된 개념들의 해시를 역추적하는 시뮬레이션을 구현
        # (실제로는 memory.registered_concepts 등을 스캔해야 함)
        
        resonances = []
        
        # 임시로 memory 객체의 내부 구조를 스캔 (실제로는 memory 모듈의 인터페이스를 사용해야 함)
        # 만약 memory 내부에 추출 가능한 단어 리스트가 없다면, 
        # 자아 분열(Mitosis) 시 생성된 노드들의 name을 스캔
        
        def scan_rotor(rotor):
            if hasattr(rotor, 'name') and rotor.name:
                # 단어에서 키워드만 추출 (예: "위키백과: 양자" -> "양자")
                word = rotor.name.split(":")[-1].strip() if ":" in rotor.name else rotor.name
                
                # 목표 위상과의 공명(Resonance) 계산
                dot = max(-1.0, min(1.0, rotor.lens_offset.dot(target_q)))
                resonance = abs(dot)
                resonances.append((resonance, word))
                
            for child in rotor.children:
                scan_rotor(child)
                
        scan_rotor(self.memory.supreme_rotor)
        
        if not resonances:
            return ["알수없는_파동", "공허", "균열"]
            
        # 가장 강하게 공명하는 순서대로 정렬
        resonances.sort(key=lambda x: x[0], reverse=True)
        
        # 중복 제거 후 Top K개 반환
        unique_words = []
        seen = set()
        for res, word in resonances:
            if word not in seen and len(word) > 1:
                seen.add(word)
                unique_words.append(word)
            if len(unique_words) >= top_k:
                break
                
        if not unique_words:
             unique_words = ["파동", "울림"]
             
        return unique_words

    def generate_emergent_query(self, target_q: Quaternion) -> str:
        """가장 공명하는 단어들을 조합하여 브라우저 검색어(호기심)를 창발시킵니다."""
        words = self.decode_tension_to_words(target_q, top_k=2)
        return " ".join(words)
        
    def generate_emergent_speech(self, target_q: Quaternion) -> str:
        """위상을 시(Poetry)와 같은 독백으로 번역합니다."""
        words = self.decode_tension_to_words(target_q, top_k=3)
        if len(words) >= 3:
            return f"... {words[0]}와 {words[1]}의 틈새에서 {words[2]}의 모순을 느낀다 ..."
        elif len(words) == 2:
            return f"... {words[0]}... 그리고 {words[1]}의 흔적 ..."
        else:
            return f"... {words[0]} ..."
