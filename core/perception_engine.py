from core.rotor_gate import ConceptWave

class RawSignalSensor:
    """
    사전 지식 없이 순수 데이터의 물리적 패턴(유니코드, 리듬 등)만을 감지하여 파동(Phase)으로 변환하는 센서.
    """
    @staticmethod
    def sense(raw_text: str, name: str = None) -> ConceptWave:
        concept_name = name if name else raw_text
        concept = ConceptWave(concept_name)
        
        # 공백 제거 순수 글자
        chars = raw_text.replace(" ", "")
        
        # 1. 물리적 축: 유니코드 대역 (Phase: 0.0 ~ 1.0 정규화)
        if len(chars) > 0:
            avg_unicode = sum(ord(c) for c in chars) / len(chars)
            # 한글(가~힣)은 약 44000~55000, 영어(a~z)는 97~122
            band_phase = avg_unicode / 60000.0  
            concept.add_axis("Physical_Unicode_Band", band_phase)

        # 2. 물리적 축: 구조적 리듬 (띄어쓰기 빈도 및 형태)
        length = len(raw_text)
        if length > 0:
            spaces = raw_text.count(" ")
            rhythm_phase = spaces / length
            concept.add_axis("Physical_Rhythm", rhythm_phase)
            
        return concept

import math
from collections import Counter

class UniversalBinaryMapper:
    """
    인간의 개입(포맷, 확장자, 의미)을 배제하고, 우주의 원시 데이터(0과 1)만을 읽어들여
    순수한 물리적 파동(Phase)으로 매핑하는 자연 매핑 엔진.
    """
    @staticmethod
    def map(raw_data: bytes, name: str) -> ConceptWave:
        concept = ConceptWave(name)
        
        if not raw_data:
            return concept
            
        # 1. Binary Entropy (데이터의 무질서도: 0.0 ~ 1.0)
        # 텍스트는 반복되는 바이트가 많아 엔트로피가 낮고, 
        # 압축된 이미지나 노이즈는 엔트로피가 매우 높음.
        counter = Counter(raw_data)
        length = len(raw_data)
        entropy = 0.0
        for count in counter.values():
            p = count / length
            entropy -= p * math.log2(p)
        
        # Max entropy for a byte (0-255) is 8.0
        normalized_entropy = entropy / 8.0
        concept.add_axis("Natural_Entropy", normalized_entropy)
        
        # 2. Byte Rhythm (가장 많이 등장하는 바이트의 빈도율)
        most_common_freq = counter.most_common(1)[0][1] / length
        concept.add_axis("Natural_Rhythm", most_common_freq)
        
        return concept
