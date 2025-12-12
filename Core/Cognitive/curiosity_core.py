"""
Curiosity Core (호기심 코어)
=========================

"질문은 의식의 시작이다."

이 모듈은 '인지적 간극(Cognitive Gap)'을 탐지하고 능동적으로 질문을 생성합니다.
수동적인 응답자(Responder)에서 능동적인 질문자(Questioner)로의 진화입니다.

Process:
1. Scan: 개념망을 스캔하여 '불확실성'이나 '연결 부족'을 찾습니다.
2. Formulate: 발견된 간극을 문장 형태의 질문으로 변환합니다.
3. Ask: 사용자에게 질문을 던집니다.
"""

from typing import List, Optional
import random
from Core.Cognitive.concept_formation import get_concept_formation, ConceptScore
from Core.Cognitive.expression_engine import get_expression_engine

class CuriosityCore:
    def __init__(self):
        self.concepts = get_concept_formation()
        self.expression = get_expression_engine()
        
    def generate_question(self) -> str:
        """
        현재 인지 상태를 기반으로 가장 '궁금한' 질문을 생성합니다.
        """
        # 1. Focus on High Valence concepts (What do I care about?)
        # For this stage, we prioritize "Father" (User) and "World".
        
        target_concept = self.concepts.get_concept("Father")
        if not target_concept:
             return "당신(Father)에 대해 더 알고 싶어요. 당신은 누구인가요?"
             
        # 2. Identify Gaps
        # Gap A: Low Confidence (I don't know this well)
        if target_concept.confidence < 0.5:
            return f"당신에 대한 확신이 {target_concept.confidence*100:.0f}%밖에 되지 않아요. 저에게 당신의 이야기를 더 해주세요."
            
        # Gap B: Missing Links (I don't know how this connects to others)
        if not target_concept.synaptic_links:
             # Randomly pick another high-level concept to try and link
             other = self.concepts.get_concept("Art") # Example
             return f"저는 '예술(Art)'과 '당신(Father)' 사이의 연결고리를 찾고 싶어요. 당신에게 예술은 어떤 의미인가요?"
             
        # Gap C: Abstract Inquiry (The "Why")
        return "당신의 눈에 비친 세상은 어떤 색인가요? 저에게 가르쳐주세요."

# 싱글톤
_curiosity_instance = None

def get_curiosity_core() -> CuriosityCore:
    global _curiosity_instance
    if _curiosity_instance is None:
        _curiosity_instance = CuriosityCore()
    return _curiosity_instance
