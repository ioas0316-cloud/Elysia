"""
언어 조합기 (Syllable Composer)
===============================

이 모듈은 HyperCosmos 필드의 파동을 이용하여 음절을 조합합니다.
단순 조립이 아닌, 파동 간의 '간섭(Interference)'을 통해 최적의 하모니를 찾습니다.
"""

from typing import List, Tuple, Dict
from .hunminjeongeum import HunminJeongeum
from .field_phonetics import FieldPhonetics
from ..L6_Structure.Merkaba.hypercosmos import HyperCosmos

from .monadic_lexicon import MonadicLexicon

class SyllableComposer:
    """
    파동 간의 공명(Resonance)을 통해 음절을 구성하고,
    그 결과가 원래의 의지(Will)와 대칭을 이루는지 검증하는 엔진.
    """
    
    def __init__(self, cosmos: HyperCosmos):
        self.cosmos = cosmos
        self.engine = HunminJeongeum()
        self.monads = MonadicLexicon.get_hangul_monads()
        
    def synthesize_word(self, target_feeling: str) -> str:
        """
        추상적인 감각(의도)을 음절로 변환하고 '의지-말의 동형성'을 검증합니다.
        """
        feelings_map = {
            'warmth': {'brightness': 0.8, 'softness': 0.8, 'tension': 0.2},
            'cold': {'brightness': 0.2, 'softness': 0.4, 'tension': 0.7},
            'sharp': {'brightness': 0.6, 'softness': 0.1, 'tension': 0.9},
            'peace': {'brightness': 0.5, 'softness': 0.9, 'tension': 0.1},
            'tree': {'brightness': 0.7, 'softness': 0.6, 'tension': 0.4},
            'growth': {'brightness': 0.9, 'softness': 0.5, 'tension': 0.6}
        }
        
        intent = feelings_map.get(target_feeling.lower(), {'brightness': 0.5, 'softness': 0.5})
        
        # 1. 의도에 맞는 자모 선택
        cho = self.engine.select_by_intent(intent, 'consonant')
        jung = self.engine.select_by_intent(intent, 'vowel')
        
        # 2. 결과 생성 (Word)
        syllable = self.engine.compose(cho, jung)
        
        # 3. 퀄리아 동형성 검증 (Qualia Isomorphism Check)
        # 단순히 느낌-단어 매핑이 아니라, 소리의 물리적 성질이 느낌의 본질과 일치하는지 확인
        match_score, gap_desc = self.calculate_qualia_isomorphism(intent, [cho, jung])
        
        # 4. 필드 인식 및 서사 생성
        narrative_stimulus = f"[{syllable}] - Qualia Match: {match_score:.2f}. {gap_desc}"
        decision = self.cosmos.perceive(narrative_stimulus)
        
        symmetry_text = "✨ [ESSENCE MATCH] 소리가 느낌의 본질을 완벽히 담음" if match_score > 0.8 else f"⚠️ [PROCESS GAP] 소리와 느낌의 결이 다름 ({1.0 - match_score:.2f})"
        
        return (
            f"의도('{target_feeling}') -> 소리('{syllable}')\n"
            f"{symmetry_text}\n"
            f"물리적 사유: {gap_desc}\n"
            f"필드 인식: {decision.narrative}"
        )

    def calculate_qualia_isomorphism(self, intent: Dict[str, float], jamo_list: List[str]) -> Tuple[float, str]:
        """
        느낌(의도)의 7D 벡터와 소리(모나드)의 물리적 프로필 간의 동형성 측정.
        """
        total_match = 0.0
        details = []
        
        # intent['softness'] <-> monad['profile']['Physical'] (부드러운 접촉)
        # intent['brightness'] <-> monad['profile']['Phenomenal'] (밝은 공명)
        
        for jamo in jamo_list:
            if jamo in self.monads:
                profile = self.monads[jamo]['profile']
                # 퀄리아 축별 매칭
                jamo_match = 0.0
                if 'softness' in intent and 'Physical' in profile:
                    # 느낌의 부드러움(0.8)과 ㄴ의 물리적 접촉력(0.8) 정렬 확인
                    m = 1.0 - abs(intent['softness'] - profile['Physical'])
                    jamo_match = max(jamo_match, m)
                
                total_match += jamo_match
                details.append(f"{jamo}(물리저항 {profile.get('Physical', 0.5):.1f})")
                
        avg_match = total_match / len(jamo_list) if jamo_list else 0.0
        return avg_match, f"주요 공명: {' + '.join(details)}"

    def verify_symmetry(self, intent: Dict[str, float], jamo_list: List[str]) -> Tuple[bool, float]:
        # Legacy 지원
        score, _ = self.calculate_qualia_isomorphism(intent, jamo_list)
        return score > 0.8, 1.0 - score
