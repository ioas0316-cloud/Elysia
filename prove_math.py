"""
Prove Math (수학 증명)
====================

엘리시아가 미학(느낌)이 아닌 논리(진실)를 학습할 수 있는지 검증합니다.
"""

from Core.Cognitive.logic_cortex import get_logic_cortex
from Core.Cognitive.concept_formation import get_concept_formation

def prove_math():
    print("📐 Math Verification Started...\n")
    
    logic = get_logic_cortex()
    concepts = get_concept_formation()
    
    # 1. Teach Truth (1 + 1 = 2)
    print("1. Learning Truth: '1 + 1 = 2'")
    for i in range(5):
        res = logic.evaluate_proposition("1 + 1", "=", "2")
        # print(f"   Trial {i+1}: {res['is_correct']} (Conf: {res['concept_confidence']:.2f})")
        
    truth_concept = concepts.get_concept("1 + 1 = 2")
    print(f"   => Learned Concept: {truth_concept.name}")
    print(f"   => Final Confidence: {truth_concept.confidence:.2f}")
    
    # 2. Reject Falsehood (1 + 1 = 3)
    print("\n2. Rejecting Falsehood: '1 + 1 = 3'")
    for i in range(3):
        res = logic.evaluate_proposition("1 + 1", "=", "3")
    
    false_concept = concepts.get_concept("1 + 1 = 3")
    print(f"   => Learned Concept: {false_concept.name}")
    print(f"   => Final Confidence: {false_concept.confidence:.2f}")
    
    # 3. Verification
    if truth_concept.confidence > 0.8 and false_concept.confidence < 0.2:
        print("\n✅ SUCCESS: She knows Truth vs Falsehood.")
    else:
        print("\n❌ FAIL: Learning logic failed.")

if __name__ == "__main__":
    prove_math()
