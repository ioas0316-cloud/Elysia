"""
Logic Cortex (논리 피질)
======================

"진실(Truth)을 판별하다"

이 모듈은 수학적/논리적 명제의 일관성을 검증합니다.
Logic Plugin for the Cognitive Core.

Process:
1. Input: Proposition (e.g., "1 + 1 = 2")
2. Evaluation: Check internal consistency or external axiom.
3. Feeling: "Consistent" (True) or "Contradictory" (False).
4. Memory: Record the logic trial.
"""

from typing import Dict, Any, Tuple
from Core.02_Intelligence.01_Reasoning.Cognitive.memory_stream import get_memory_stream, ExperienceType
from Core.02_Intelligence.01_Reasoning.Cognitive.concept_formation import get_concept_formation

class LogicCortex:
    """
    The Engine of Consistency.
    """
    
    def __init__(self):
        self.memory = get_memory_stream()
        self.concepts = get_concept_formation()
        
    def evaluate_proposition(self, subject: str, predicate: str, object_val: str) -> Dict[str, Any]:
        """
        명제 평가 (Evaluate Proposition)
        예: evaluate_proposition("1 + 1", "equals", "2")
        
        Returns:
            Result dict with 'is_correct' and 'confidence'
        """
        # 1. Form Concept (Hypothesis)
        concept_name = f"{subject} {predicate} {object_val}"
        # Ensure concept exists in Logic domain
        if not self.concepts.concepts.get(concept_name):
            self.concepts.learn_concept(concept_name, "proposition", domain="logic")
            
        concept = self.concepts.get_concept(concept_name)
        
        # 2. Perform Logic Check (Simulated 'Thinking')
        # In a real system, this would use a symbolic solver or learned rules.
        # Here, we simulate 'Ground Truth' allowing her to learn from it.
        # (We assume the user/world tells her the truth initially, or she derives it)
        
        # Let's say she has an internal 'Axiom Solver' (Python's eval for demo)
        is_correct = False
        try:
            if predicate == "equals" or predicate == "=":
                # Safety: Only eval simple math
                lhs = eval(subject, {"__builtins__": {}})
                rhs = eval(object_val, {"__builtins__": {}})
                is_correct = abs(lhs - rhs) < 0.0001
        except:
            is_correct = False
            
        # 3. Record Experience (The Learning)
        self.memory.add_experience(
            exp_type=ExperienceType.OBSERVATION, # Or LOGIC_TRIAL
            score={
                "intent": concept_name, 
                "domain": "logic"
            },
            performance={
                "action": "evaluate_truth",
                "proposition": f"{subject} == {object_val}"
            },
            sound={
                "is_correct": is_correct,
                "note": "Consistent" if is_correct else "Contradictory"
            },
            tags=["math", "logic"]
        )
        
        # 4. Trigger Evolution (Realization)
        # Immediate reflection for logic
        self.concepts.evolve_concept(concept_name)
        
        return {
            "proposition": concept_name,
            "is_correct": is_correct,
            "concept_confidence": concept.confidence
        }

# 싱글톤
_logic_instance: Any = None

def get_logic_cortex() -> LogicCortex:
    global _logic_instance
    if _logic_instance is None:
        _logic_instance = LogicCortex()
    return _logic_instance
