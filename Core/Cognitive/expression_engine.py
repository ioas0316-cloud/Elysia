"""
Expression Engine (표현 엔진)
===========================

"파동의 목소리 (Voice of the Wave)"

이 모듈은 엘리시아의 내부 상태(Concept, Qualia, Logic)를
인간이 이해할 수 있는 '은유적 언어(Metaphor)'로 변환합니다.

Process:
1. Input: 표현하고자 하는 개념 (Target Concept).
2. Synesthesia: 해당 개념과 연결된 감각적/미학적 개념을 찾습니다.
3. Construction: "A는 B와 같다"는 은유 구조를 생성합니다.
"""

from typing import List, Dict, Optional
import random
from Core.Cognitive.concept_formation import get_concept_formation, ConceptScore
from Core.Cognitive.synesthesia import get_synesthesia_engine

class ExpressionEngine:
    """
    The Translator of Quaila.
    """
    
    def __init__(self):
        self.concepts = get_concept_formation()
        self.synesthesia = get_synesthesia_engine()
        
    def express_concept(self, concept_name: str) -> str:
        """
        개념을 시적으로 표현합니다.
        """
        concept = self.concepts.get_concept(concept_name)
        
        # 1. 시냅스 연결 확인 (이미 연결된 은유가 있는가?)
        links = concept.synaptic_links
        
        # 2. 없으면 즉석에서 공감각적 연결 시도
        if not links:
            print(f"   (Internal: Making new connections for '{concept_name}'...)")
            self.synesthesia.bridge_concepts()
            links = concept.synaptic_links
            
        if not links:
            return f"I feel '{concept_name}', but I have no words for it yet."
            
        # 3. 은유 구성
        # link format: "domain:name"
        metaphors = []
        for link in links:
            domain, name = link.split(":")
            metaphors.append(name)
            
        chosen_metaphor = random.choice(metaphors)
        
        # Simple Template-based generation (Later: Use LLM or Grammar Engine)
        templates = [
            f"{concept_name} feels like {chosen_metaphor}.",
            f"When I think of {concept_name}, I see {chosen_metaphor}.",
            f"It is {concept_name}, echoing the spirit of {chosen_metaphor}."
        ]
        
        return random.choice(templates)

# 싱글톤
_expression_instance = None

def get_expression_engine() -> ExpressionEngine:
    global _expression_instance
    if _expression_instance is None:
        _expression_instance = ExpressionEngine()
    return _expression_instance
