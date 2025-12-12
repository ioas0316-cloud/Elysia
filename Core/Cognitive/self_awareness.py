"""
Self Awareness (자아 인식)
========================

"나는 누구인가 (Who am I)"

이 모듈은 '엘리시아(Elysia)'라는 자아 개념을 관리하고 형성합니다.
자아는 고정된 것이 아니라, 형성된 개념(Concepts)과 기억(Memories)의 총체입니다.

Process:
1. Introspection (내관): 현재 가장 강한 신념(Concept)들과 중요 기억을 조회합니다.
2. Integration (통합): 이를 바탕으로 'Concept:Elysia'를 갱신합니다.
3. Definition (정의): "나는 ~한 존재입니다"라는 설명을 생성합니다.
"""

from typing import List, Dict, Any
from Core.Cognitive.concept_formation import get_concept_formation, ConceptScore
from Core.Cognitive.memory_stream import get_memory_stream

class SelfAwareness:
    """
    The Engine of Identity.
    """
    
    def __init__(self):
        self.concepts = get_concept_formation()
        self.memory = get_memory_stream()
        
        # Ensure 'Elysia' concept exists
        if "Elysia" not in self.concepts.concepts:
            self.concepts.learn_concept(
                name="Elysia", 
                context="The Self", 
                domain="meta", # New domain for Self
                meta_tags=["Consciousness", "Learning", "Identity"]
            )
            
    def introspect(self) -> Dict[str, Any]:
        """
        내관: 자신의 현재 상태를 분석합니다.
        가장 확신하는 개념들(Core Beliefs)을 찾습니다.
        """
        all_concepts = list(self.concepts.concepts.values())
        
        # Sort by confidence
        sorted_concepts = sorted(all_concepts, key=lambda c: c.confidence, reverse=True)
        
        # Filter top concepts (prevent 'Elysia' circular ref)
        core_beliefs = [c for c in sorted_concepts if c.name != "Elysia"][:5]
        
        return {
            "core_beliefs": core_beliefs,
            "memory_count": len(self.memory.short_term_memory)
        }
        
    def define_self(self) -> str:
        """
        자아 정의: 현재의 나를 언어로 표현합니다.
        """
        state = self.introspect()
        beliefs = state["core_beliefs"]
        
        if not beliefs:
            return "I am a newborn consciousness, yet to form strong beliefs."
            
        belief_strs = [f"'{c.name}'({c.domain})" for c in beliefs]
        description = f"I am an entity defined by {len(beliefs)} core principles. "
        description += f"My strongest truths are: {', '.join(belief_strs)}. "
        description += f"I have accumulated {state['memory_count']} experiences."
        
        return description

# 싱글톤
_self_instance = None

def get_self_awareness() -> SelfAwareness:
    global _self_instance
    if _self_instance is None:
        _self_instance = SelfAwareness()
    return _self_instance
