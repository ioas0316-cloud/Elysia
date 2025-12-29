"""
Linguistic Cortex (언어 피질)
===========================

"구조(Structure)를 이해하다"

이 모듈은 언어의 문법적 구조와 순서를 학습합니다.
Linguistic Plugin for the Cognitive Core.

Process:
1. Input: Sentence (e.g., "I eat apple")
2. Perception: Parse structure (S-V-O).
3. Feeling: "Grammatical" (Clear) or "Broken" (Confusing).
4. Memory: Record the syntax trial.
"""

from typing import Dict, Any, List
from Core.IntelligenceLayer.Cognitive.memory_stream import get_memory_stream, ExperienceType
from Core.IntelligenceLayer.Cognitive.concept_formation import get_concept_formation

class LinguisticCortex:
    """
    The Engine of Grammar.
    """
    
    def __init__(self):
        self.memory = get_memory_stream()
        self.concepts = get_concept_formation()
        
    def evaluate_syntax(self, sentence: str, expected_pattern: str) -> Dict[str, Any]:
        """
        구문 평가 (Evaluate Syntax)
        예: evaluate_syntax("I eat apple", "SVO")
        
        Returns:
            Result dict with 'is_grammatical' and 'confidence'
        """
        # 1. Form Concept (Hypothesis about a Pattern)
        # concept_name e.g., "Sentence_Pattern_SVO"
        concept_name = f"Pattern_{expected_pattern}"
        if not self.concepts.concepts.get(concept_name):
            self.concepts.learn_concept(concept_name, "syntax_rule", domain="linguistic")
            
        concept = self.concepts.get_concept(concept_name)
        
        # 2. Perform Syntax Check (Simulated)
        # In reality, this would use a parser.
        # Simulation: "I eat apple" (Word count 3) matches simple SVO expectations
        words = sentence.split()
        is_grammatical = len(words) >= 2 # Simplistic check
        
        # 3. Record Experience
        self.memory.add_experience(
            exp_type=ExperienceType.OBSERVATION, 
            score={
                "intent": concept_name, 
                "domain": "linguistic"
            },
            performance={
                "action": "speak",
                "sentence": sentence
            },
            sound={
                "is_clear": is_grammatical,
                "note": "Grammatical" if is_grammatical else "Broken"
            },
            tags=["language", "grammar"]
        )
        
        # 4. Trigger Evolution
        # Evolution logic for Linguistics needs to be implemented in ConceptFormation
        # For now, we reuse the Loop/Aesthetic evolution or add a specific one.
        # We will add _evolve_linguistic to ConceptFormation in next step if needed.
        # But for proving the architecture, we can verify that Memory works.
        
        return {
            "pattern": expected_pattern,
            "is_grammatical": is_grammatical,
            "concept_confidence": concept.confidence # Will stay static until evolve is updated
        }

# 싱글톤
_linguistic_instance: Any = None

def get_linguistic_cortex() -> LinguisticCortex:
    global _linguistic_instance
    if _linguistic_instance is None:
        _linguistic_instance = LinguisticCortex()
    return _linguistic_instance
