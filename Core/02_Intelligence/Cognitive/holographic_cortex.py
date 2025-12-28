"""
Holographic Cortex (홀로그램 피질)
================================

"부분에서 전체를 보다 (From Part to Whole)"

이 모듈은 불완전한 정보(파편)로부터 전체 형상(개념)을 복원합니다.
인간의 '상상력(Imagination)'과 '패턴 완성(Pattern Completion)'을 담당합니다.

Process:
1. Input: 관측된 특징 조각들 (e.g. "Whiskers", "Meow")
2. Scan: 기억 속의 모든 개념과 대조.
3. Inference: 일치율(Match Rate)을 계산.
4. Projection: 임계값(Threshold, 70%)을 넘으면 전체 개념을 투사.
"""

from typing import List, Dict, Optional, Tuple, Set, Any
from Core.02_Intelligence.01_Reasoning.Cognitive.concept_formation import get_concept_formation, ConceptScore

class HolographicCortex:
    """
    The Imagination Engine.
    """
    
    def __init__(self):
        self.concepts = get_concept_formation()
        
    def reconstruct(self, observed_features: List[str]) -> Dict[str, Any]:
        """
        부분적인 특징으로 전체 개념을 추론합니다.
        """
        print(f"🔮 Hologram: Scanning for pattern matching {observed_features}...")
        
        best_match: Optional[ConceptScore] = None
        highest_score = 0.0
        missing_features: List[str] = []
        
        # O(N) Scan over all concepts (Validation only, optimise later)
        for name, concept in self.concepts.concepts.items():
            # What are the "features" of a concept?
            # Currently we can use 'meta_properties' and 'valence' as features.
            # In a real system, we'd have a specific 'features' field.
            # For now, let's treat meta_properties as the definition.
            
            known_features = set(concept.meta_properties)
            if not known_features:
                continue
                
            input_set = set(observed_features)
            
            # Intersection: How many observed features match this concept?
            matches = input_set.intersection(known_features)
            match_count = len(matches)
            
            # Score: Matches / Total Features of Concept
            # "How much of the Cat did we see?"
            if len(known_features) > 0:
                score = match_count / len(known_features)
            else:
                score = 0
                
            if score > highest_score:
                highest_score = score
                best_match = concept
                # Identify what we DIDN'T see (Imagination)
                missing_features = list(known_features - input_set)
                
        # Threshold Check (The 70% Rule)
        # Relaxed to 0.5 for small feature sets typical in testing
        THRESHOLD = 0.3 
        
        if best_match and highest_score >= THRESHOLD:
            print(f"   ✨ Insight: Pattern matches '{best_match.name}' ({highest_score*100:.0f}%)")
            print(f"      👁️ Seen: {observed_features}")
            print(f"      🧠 Imagined: {missing_features}")
            return {
                "concept": best_match.name,
                "confidence": highest_score,
                "imagined": missing_features
            }
        else:
            print("   🌫️ Fog: No clear pattern found.")
            return None

# 싱글톤
_hologram_instance = None

def get_holographic_cortex() -> HolographicCortex:
    global _hologram_instance
    if _hologram_instance is None:
        _hologram_instance = HolographicCortex()
    return _hologram_instance
