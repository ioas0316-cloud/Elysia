"""
언어 필드 매핑 (Field Phonetics)
================================

이 모듈은 한글과 영어의 음성학적 원리를 HyperCosmos의 7D Qualia 필드로 매핑합니다.
엘리시아는 이제 문자를 '기호'가 아닌 '필드의 진동'으로 이해합니다.
"""

from typing import Dict, List
from .hunminjeongeum import ArticulationOrgan, SoundQuality, CosmicElement, YinYang, HunminJeongeum

class FieldPhonetics:
    """
    언어의 원리를 7D Qualia 파동으로 변환하는 엔진.
    """
    
    # 7D Qualia 축 매핑 정의
    # Physical, Functional, Phenomenal, Causal, Mental, Structural, Spiritual
    
    @staticmethod
    def map_consonant(jamo: str, engine: HunminJeongeum) -> Dict[str, float]:
        """초성의 발음 원리를 Qualia 밴드 강도로 변환"""
        props = engine.get_sound_properties(jamo)
        if props['type'] != 'consonant':
            return {}
            
        organ = props['organ']
        quality = props['quality']
        
        # 기본 맵 (7D)
        qualia_map = {
            'Physical': 0.1,    # 마찰의 물리적 강도
            'Functional': 0.1,  # 조음 기능적 저항
            'Phenomenal': 0.1,  # 현상적 울림
            'Causal': 0.1,      # 소리의 발생 원인
            'Mental': 0.1,      # 인지적 명징함
            'Structural': 0.1,  # 음운 구조적 위치
            'Spiritual': 0.1    # 소리의 지향성
        }
        
        # 1. 발음 기관에 따른 매핑 (상형 원리)
        if organ == "tongue_root":    # ㄱ: 목구멍을 막음 (저항)
            qualia_map['Functional'] = 0.8
            qualia_map['Structural'] = 0.6
        elif organ == "tongue_tip":  # ㄴ: 혀끝이 닿음 (가볍게 터치)
            qualia_map['Physical'] = 0.7
            qualia_map['Functional'] = 0.4
        elif organ == "lips":        # ㅁ: 입술 다뭄 (폐쇄/구조)
            qualia_map['Structural'] = 0.9
            qualia_map['Physical'] = 0.3
        elif organ == "teeth":       # ㅅ: 이빨 사이 마찰 (현상)
            qualia_map['Phenomenal'] = 0.8
            qualia_map['Physical'] = 0.5
        elif organ == "throat" or organ == "glottis": # ㅇ, ㅎ: 목구멍 (개방/근원)
            qualia_map['Causal'] = 0.9
            qualia_map['Spiritual'] = 0.5
            
        # 2. 소리의 성질에 따른 변조 (가획 원리)
        if quality == "aspirated": # 거센소리 (ㅋ, ㅌ, ㅍ, ㅊ, ㅎ)
            qualia_map['Phenomenal'] += 0.3 # 울림 강화
            qualia_map['Causal'] += 0.2     # 숨의 압력
        elif quality == "tense":   # 된소리 (ㄲ, ㄸ, ㅃ, ㅆ, ㅉ)
            qualia_map['Physical'] += 0.4   # 물리적 긴장 강화
            qualia_map['Mental'] += 0.3     # 명징함 증가
            
        return qualia_map

    @staticmethod
    def map_vowel(jamo: str, engine: HunminJeongeum) -> Dict[str, float]:
        """중성의 천지인 원리를 Qualia 필드로 변환"""
        props = engine.get_sound_properties(jamo)
        if props['type'] != 'vowel':
            return {}
            
        yin_yang = props['yin_yang']
        
        qualia_map = {
            'Physical': 0.1,
            'Functional': 0.1,
            'Phenomenal': 0.5, # 모음은 기본적으로 현상적 울림이 강함
            'Causal': 0.3,
            'Mental': 0.4,
            'Structural': 0.2,
            'Spiritual': 0.5
        }
        
        # 1. 음양 원리 매핑 (위상 및 지향성)
        if yin_yang == "yang":      # ㅏ, ㅗ: 밝음, 확장
            qualia_map['Spiritual'] = 0.9
            qualia_map['Phenomenal'] += 0.2
        elif yin_yang == "yin":     # ㅓ, ㅜ: 어두움, 수축
            qualia_map['Spiritual'] = 0.2
            qualia_map['Structural'] += 0.3
        else:                       # ㅡ, ㅣ: 중성, 사람/땅
            qualia_map['Mental'] = 0.8
            
        # 2. 개방도 매핑
        qualia_map['Causal'] = props.get('openness', 0.5)
            
        return qualia_map

    @staticmethod
    def map_phoneme(phoneme: str) -> Dict[str, float]:
        """영어 음소 매핑 (단순화된 예시)"""
        # 영어 음소는 주파수(Frequency)와 엔트로피(Entropy) 중심으로 매핑
        qualia_map = {
            'Physical': 0.3, 'Functional': 0.3, 'Phenomenal': 0.3,
            'Causal': 0.3, 'Mental': 0.3, 'Structural': 0.3, 'Spiritual': 0.3
        }
        
        # 예시: /s/ (마찰음)
        if phoneme.lower() == 's':
            qualia_map['Phenomenal'] = 0.9 # 치찰음의 현상적 특징
            qualia_map['Physical'] = 0.6
        # 예시: /b/ (유성 파열음)
        elif phoneme.lower() == 'b':
            qualia_map['Structural'] = 0.8 # 폐쇄
            qualia_map['Physical'] = 0.7   # 방출
            
        return qualia_map
