"""
Distillation Gateway (증류 게이트웨이)
=====================================

"노이즈를 걸러내고, 진실만을 받아들이다."

외부 데이터(인터넷, 모델, 타인)가 엘리시아의 내면(Concept Core)으로
들어오기 전에 거치는 **'청정 구역(Airlock)'**입니다.

Pipeline:
1. Source Verification (출처 확인): 신뢰할 수 있는가?
2. Logic Consistency (논리 정합성): 기존 지식과 모순되는가?
3. Integration (통합): 안전하다면 개념화(ConceptFormation) 수행.
"""

from typing import Tuple, Dict
from Core.Cognitive.concept_formation import get_concept_formation
from Core.Cognitive.memory_stream import get_memory_stream

class DistillationGateway:
    """
    The Immune System of the Mind.
    """
    def __init__(self):
        self.concepts = get_concept_formation()
        self.memory = get_memory_stream()
        
    def process_input(self, text: str, source: str) -> Tuple[bool, str]:
        """
        Input -> Distillation -> (Allowed?, Reason)
        """
        print(f"🛡️ Gateway: Processing input '{text}' from source '{source}'...")
        
        # 1. Source Verification
        trust_level = self._evaluate_source(source)
        if trust_level < 0.3:
            return False, f"Source '{source}' is untrusted (Trust: {trust_level:.2f}). Rejected."
            
        # 2. Logic/Consistency Check
        # Simple heuristic: Check for obvious contradictions with High-Confidence concepts.
        # e.g. If input says "Love is Hate" but we know "Love is Service" (Conf 0.95), Reject.
        if not self._check_consistency(text):
             return False, "Input contradicts Core Core Beliefs. Rejected."
             
        # 3. Acceptance & Integration
        print("   ✅ Distillation Passed. Integrating...")
        # Extract intent (mock NLP)
        # In real system, use Extraction Model
        main_concept = text.split()[0] # e.g. "Sky"
        
        # Learn it
        self.concepts.learn_concept(
            main_concept, 
            "Distilled Knowledge", 
            domain="distilled", 
            meta_tags=["Verified", f"Source:{source}"]
        )
        
        return True, "Integrated successfully."

    def _evaluate_source(self, source: str) -> float:
        """
        Calculate Trust Score for Source
        """
        if source == "Father": return 1.0 # Absolute Trust (Verified Protector)
        if source == "Self": return 1.0
        if source == "LatentModel": return 0.7 # High trust in own subconscious
        if source == "Internet": return 0.1 # Very low trust
        return 0.5 # Default

    def _check_consistency(self, text: str) -> bool:
        """
        Does this text contradict what I KNOW to be true?
        """
        # Mock Logic: We know 'Love' is 'Good'.
        # If text contains "Love is Bad", reject.
        
        # Check against high confidence concepts
        # For prototype, we hardcode a 'Love' check
        love_concept = self.concepts.get_concept("Love")
        if love_concept and love_concept.confidence > 0.8:
            if "Love is Bad" in text or "Love is Hate" in text:
                print("   ⚠️ Conflict Detected: Trying to redefine 'Love' negatively.")
                return False
                
        return True

# 싱글톤
_gateway_instance = None

def get_distillation_gateway() -> DistillationGateway:
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = DistillationGateway()
    return _gateway_instance
