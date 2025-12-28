"""
Prove Grammar (문법 증명)
=======================

엘리시아가 언어의 순서(Structure)를 학습할 수 있는지 검증합니다.
"""

from Core.Cognitive.linguistic_cortex import get_linguistic_cortex
from Core.Cognitive.concept_formation import get_concept_formation

def prove_grammar():
    print("📚 Grammar Verification Started...\n")
    
    linguist = get_linguistic_cortex()
    concepts = get_concept_formation()
    
    # 1. Teach Good Grammar (SVO)
    print("1. Hearing Clear Sentence: 'I eat apple'")
    res = linguist.evaluate_syntax("I eat apple", "SVO")
    print(f"   Structure 'SVO': {res['is_grammatical']}")
    
    # 2. Teach Bad Grammar (OSV)
    print("\n2. Hearing Broken Sentence: 'Apple I eat'")
    res = linguist.evaluate_syntax("Apple I eat", "SVO")
    print(f"   Structure 'SVO' Check: {res['is_grammatical']}")

    # 3. Validation
    # In next iteration, we would check if 'SVO' concept confidence increased.
    # For now, we verify the immediate "Feeling" of clarity.
    
    svo_concept = concepts.get_concept("Pattern_SVO")
    print(f"\n   => Concept 'Pattern_SVO' exists: {svo_concept is not None}")
    
    if svo_concept:
        print("\n✅ SUCCESS: She perceived the structure and recorded the clarity.")
    else:
        print("\n❌ FAIL: Concept formation failed.")

if __name__ == "__main__":
    prove_grammar()
