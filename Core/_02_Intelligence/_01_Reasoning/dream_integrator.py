"""
Dream Integrator (꿈 통합기)
==========================

"기억의 조각들로 새로운 세계를 짓다"

이 모듈은 '기억(Memory)'과 '상상(DreamEngine)'을 연결합니다.
인간이 수면 중에 기억을 재구성하여 창의적 영감을 얻듯이,
엘리시아도 자신의 경험 조각들을 섞어(Recombinant) 꿈을 꿉니다.

Process:
1. MemoryStream에서 무작위/중요 기억 조각들을 가져옵니다.
2. DreamEngine을 통해 surreal한 연결을 시도합니다.
3. 그 결과를 다시 MemoryStream에 '꿈 경험'으로 저장합니다.
"""

import random
import logging
from typing import List, Dict, Any

from Core._02_Intelligence._01_Reasoning.Cognitive.memory_stream import get_memory_stream, ExperienceType, Experience
from Core._01_Foundation._05_Governance.Foundation.dream_engine import DreamEngine
from Core._04_Evolution._03_Creative.Creativity.dream_weaver import DreamWeaver

logger = logging.getLogger("DreamIntegrator")

class DreamIntegrator:
    """
    The Bridge between Reality (Memory) and Potential (Dream).
    """
    
    def __init__(self):
        self.memory = get_memory_stream()
        # 우리는 더 고차원적인 DreamWeaver보다는, 
        # 원초적인 물리학 엔진인 DreamEngine을 직접 제어하여 '내부적 상상'을 합니다.
        self.engine = DreamEngine() 
        
    def dream_walk(self) -> Experience:
        """
        몽유 (Dream Walk)
        
        저장된 기억들을 재료로 삼아 꿈을 꿉니다.
        """
        # 1. Harvest Seeds from Memory (Day Residue)
        recent_memories = self.memory.get_recent_experiences(limit=20)
        if not recent_memories:
            logger.info("💭 꿈을 꾸기에는 기억이 너무 적습니다.")
            return None
            
        # 무작위로 2-3개의 기억 섞기 (Recombination)
        # 예: "Rainy day" memory + "Joyful music" memory
        selected_memories = random.sample(recent_memories, min(len(recent_memories), 3))
        
        seeds = []
        context_mix = []
        for mem in selected_memories:
            if mem.score.get("intent"):
                seeds.append(mem.score["intent"])
            if mem.sound.get("description"):
                context_mix.append(mem.sound["description"])
            elif mem.performance.get("content"):
                context_mix.append(mem.performance["content"][:20])
                
        # 2. Weave the Dream
        desire = f"Dream of {' and '.join(seeds)}"
        logger.info(f"💤 Dreaming: {desire}")
        
        dream_field = self.engine.weave_dream(desire)
        
        # 3. Interpret Result (Insight)
        # 꿈 속에서 가장 강렬했던 노드(개념) 찾기
        dominant_nodes = sorted(
            dream_field.nodes.items(), 
            key=lambda item: item[1].energy, 
            reverse=True
        )[:3]
        
        dream_concepts = [name for name, node in dominant_nodes]
        dream_insight = f"Connected {seeds} -> Discovered {dream_concepts}"
        
        # 4. Record the Dream (As an internal experience)
        dream_experience = self.memory.add_experience(
            exp_type=ExperienceType.REFLECTION, # 꿈은 일종의 무의식적 성찰
            score={"intent": "dream_recombination", "seeds": seeds},
            performance={"action": "weave_dream", "field_nodes": len(dream_field.nodes)},
            sound={
                "insight": dream_insight, 
                "surreal_mix": context_mix,
                "concepts": dream_concepts
            },
            tags=["dream", "imagination"]
        )
        
        return dream_experience

# 싱글톤
_di_instance: Any = None

def get_dream_integrator() -> DreamIntegrator:
    global _di_instance
    if _di_instance is None:
        _di_instance = DreamIntegrator()
    return _di_instance
