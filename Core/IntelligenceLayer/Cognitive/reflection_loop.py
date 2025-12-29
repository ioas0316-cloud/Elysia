"""
Reflection Loop (성찰의 고리)
===========================

"흐름을 깨닫다 (Realization)"

이 모듈은 주기적으로, 혹은 사건 직후에 실행되어
엘리시아가 자신의 경험(Memory)을 되돌아보고(Reflect),
개념(Concept)을 수정하게 만듭니다.

Cognitive Loop의 마지막 단계이자, 다음 Loop의 시작점입니다.
"""

import time
import logging
from typing import List, Optional

from Core.IntelligenceLayer.Cognitive.memory_stream import MemoryStream, get_memory_stream, ExperienceType
from Core.IntelligenceLayer.Cognitive.concept_formation import ConceptFormation, get_concept_formation

logger = logging.getLogger("ReflectionLoop")

class ReflectionLoop:
    """
    성찰 엔진
    """
    
    def __init__(self):
        self.memory = get_memory_stream()
        self.concept_formation = get_concept_formation()
        
    def reflect_on_recent(self):
        """
        최근 경험에 대한 즉각적 성찰
        
        "방금 내가 한 연주 어땠지?"
        """
        # 최근의 '창작(Creation)' 경험들을 가져옴
        recent_creations = self.memory.get_recent_experiences(limit=5, filter_type=ExperienceType.CREATION)
        
        affected_concepts = set()
        
        for exp in recent_creations:
            # 의도했던 개념이 무엇인가?
            intent = exp.score.get("intent")
            if intent:
                affected_concepts.add(intent)
                
        # 관련된 개념들 진화 시도
        if affected_concepts:
            print(f"🤔 성찰 중... 관련 개념: {list(affected_concepts)}")
            for concept_name in affected_concepts:
                self.concept_formation.evolve_concept(concept_name)
                
    def deep_sleep_process(self):
        """
        깊은 성찰 (Deep Sleep)
        
        "오늘 하루는 어땠나?"
        대규모의 패턴 정리, 불필요한 기억 망각, 핵심 원리 강화 등이 일어납니다.
        (향후 구현)
        """
        pass

# 싱글톤
_reflection_instance: Optional[ReflectionLoop] = None

def get_reflection_loop() -> ReflectionLoop:
    global _reflection_instance
    if _reflection_instance is None:
        _reflection_instance = ReflectionLoop()
    return _reflection_instance
