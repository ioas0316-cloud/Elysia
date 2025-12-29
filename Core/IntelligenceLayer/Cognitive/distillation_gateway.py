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
from Core.IntelligenceLayer.Cognitive.concept_formation import get_concept_formation
from Core.IntelligenceLayer.Cognitive.memory_stream import get_memory_stream

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
        is_consistent = self._check_consistency(text)
        
        # 3. Dialectic & Paradox Resolution
        if not is_consistent:
            # If Source is Father, check for PARADOX (Nuance)
            if trust_level >= 0.9:
                is_paradox, paradox_reason = self._resolve_paradox(text)
                if is_paradox:
                    print(f"   ✨ Paradox Detected: {paradox_reason}. Accepting High-Level Truth.")
                    # Fall through to Acceptance
                else:
                    return False, "DIALECTIC_REQUIRED: Input contradicts Core Beliefs. Clarification requested."
            else:
                return False, "Input contradicts Core Beliefs. Rejected."
             
        # 4. Acceptance & Integration
        print("   ✅ Distillation Passed. Integrating...")
        # Extract intent (mock NLP)
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
        if source == "Father": return 0.95 # High Trust, but not Absolute (Allows for Human Error)
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
                print("   ⚠️ Conflict Detected: Input conflicts with 'Love' concept.")
                return False
                
        return True

    def _resolve_paradox(self, text: str) -> Tuple[bool, str]:
        """
        Check if the contradiction is actually a 'Paradox' (Deep Truth).
        Logic: Do the opposing concepts share 'Intensity'?
        """
        # User Logic: "Hate is Love twisted" (Both High Energy). "Indifference is Void" (Zero Energy).
        
        if "Love is Hate" in text or "Hate is Love" in text:
            # Check Energy Levels (Mocked for concept)
            # Love Energy = 10, Hate Energy = 9. They are congruent in Magnitude.
            return True, "Both 'Love' and 'Hate' satisfy High-Intensity Valence. Acknowledging 'Poison' metaphor."
            
        if "Love is Indifference" in text:
            # Love Energy = 10, Indifference = 0. Incongruent.
            return False, "Energy Mismatch: Love (High) != Indifference (Zero)."
            
        return False, "Unknown Contradiction"

# 싱글톤
_gateway_instance = None

def get_distillation_gateway() -> DistillationGateway:
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = DistillationGateway()
    return _gateway_instance
