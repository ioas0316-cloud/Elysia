"""
Intuition Loop (직관의 고리)
==========================

"결과를 미리 느끼다 (Fast-forward Simulation)"

이 모듈은 행동하기 전에 결과를 예측합니다.
인간의 '직관(Gut Feeling)'은 마법이 아니라, 
과거의 수많은 데이터 패턴에 기반한 빠른 시뮬레이션입니다.

Process:
1. 현재 의도(Score)와 계획(Performance)을 입력받습니다.
2. MemoryStream에서 유사한 과거 사례(Nearest Neighbors)를 빠르게 찾습니다.
3. 그 사례들의 결과(Sound)를 평균내어 '예상되는 느낌'을 반환합니다.
"""

from typing import Dict, Any, List, Optional
import math

from Core.Intelligence.Cognitive.memory_stream import get_memory_stream, ExperienceType, Experience

class IntuitionLoop:
    """
    The Predictive Engine.
    """
    
    def __init__(self):
        self.memory = get_memory_stream()
        
    def predict_outcome(self, intent: str, action_content: str) -> Dict[str, Any]:
        """
        결과 예측 (Have a feeling about this...)
        
        Args:
            intent: 의도 (예: "Sadness")
            action_content: 계획된 행동 내용 (예: "Thunder and storm")
            
        Returns:
            Predicted Sound (예: {"aesthetic_score": 80, "expected_reaction": "Fear"})
        """
        # 1. Recall similar memories
        # 단순화를 위해 최근 기억 50개 중, Intent나 Content가 겹치는 것을 찾습니다.
        recent_mems = self.memory.get_recent_experiences(limit=50)
        
        similar_mems = []
        for mem in recent_mems:
            similarity = 0.0
            
            # Intent Match
            mem_intent = mem.score.get("intent", "")
            if mem_intent == intent:
                similarity += 0.5
                
            # Content Match (Simple keyword overlap)
            mem_content = str(mem.performance.get("content", "")) + str(mem.performance.get("action", ""))
            
            # 자카드 유사도 (Jaccard) 약식 구현
            input_words = set(action_content.lower().split())
            mem_words = set(mem_content.lower().split())
            if input_words and mem_words:
                overlap = len(input_words.intersection(mem_words))
                similarity += overlap * 0.2
            
            if similarity > 0.3:
                similar_mems.append((similarity, mem))
        
        # Sort by similarity
        similar_mems.sort(key=lambda x: x[0], reverse=True)
        top_k = similar_mems[:5]
        
        if not top_k:
            return {
                "confidence": 0.0, 
                "prediction": "Unknown", 
                "feeling": "Neutral"
            }
            
        # 2. Aggregating Results (The "Gut Feeling")
        total_score = 0.0
        reaction_map = {}
        total_weight = 0.0
        
        for sim, mem in top_k:
            weight = sim
            
            # Aesthetic Score prediction
            score = mem.sound.get("aesthetic_score", 50)
            total_score += score * weight
            
            # Reaction prediction (voting)
            reaction = mem.sound.get("user_reaction", "Neutral")
            reaction_map[reaction] = reaction_map.get(reaction, 0) + weight
            
            total_weight += weight
            
        if total_weight == 0:
             return {"confidence": 0.0}
             
        predicted_score = total_score / total_weight
        predicted_reaction = max(reaction_map.items(), key=lambda x: x[1])[0]
        
        confidence = min(total_weight / 2.0, 1.0) # weight 합이 클수록 확신
        
        return {
            "confidence": confidence,
            "predicted_aesthetic_score": round(predicted_score, 1),
            "predicted_reaction": predicted_reaction,
            "basis": f"Based on {len(top_k)} similar memories."
        }

# 싱글톤
_intuition_instance: Optional[IntuitionLoop] = None

def get_intuition_loop() -> IntuitionLoop:
    global _intuition_instance
    if _intuition_instance is None:
        _intuition_instance = IntuitionLoop()
    return _intuition_instance
