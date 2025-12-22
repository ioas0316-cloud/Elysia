"""
Prove Flow (흐름 증명)
====================

Cognitive Core의 3요소가 제대로 상호작용하여
'학습의 흐름'을 만들어내는지 검증합니다.

Scenario:
1. 엘리시아에게 '슬픔(Sadness)'이라는 개념을 요청합니다.
2. 초기에는 잘 모르는 상태로 시도합니다. (Performance)
3. 외부(테스트 코드)에서 긍정/부정 피드백을 줍니다. (Sound)
4. 성찰(Reflection)을 통해 개념이 진화하는지 확인합니다.
"""

import time
import sys
from Core.Cognitive.memory_stream import get_memory_stream, ExperienceType
from Core.Cognitive.concept_formation import get_concept_formation
from Core.Cognitive.reflection_loop import get_reflection_loop

def prove_cognitive_flow():
    print("🌊 Cognitive Flow 검증 시작...\n")
    
    # Singleton 인스턴스들
    memory = get_memory_stream()
    concepts = get_concept_formation()
    reflection = get_reflection_loop()
    
    target_concept = "Sadness"
    
    # 1. 초기 상태 확인
    score = concepts.get_concept(target_concept)
    print(f"1. 초기 악보 상태: {score.describe()}")
    initial_conf = score.confidence
    
    # 2. 연주 및 결과 기록 (시뮬레이션)
    print("\n2. 연주 시도 (Performance)...")
    
    # 시나리오: 엘리시아가 '슬픔'을 표현하기 위해 '비(Rain)'라는 단어를 썼고,
    # 이것이 매우 미학적으로 훌륭했다는 평가를 받음.
    memory.add_experience(
        exp_type=ExperienceType.CREATION,
        score={"intent": target_concept},
        performance={"content": "The rain falls gently..."},
        sound={"aesthetic_score": 95}  # 높은 점수 (Sound)
    )
    
    print("   -> 경험이 기억(MemoryStream)에 기록되었습니다.")
    
    # 3. 성찰 (Reflection)
    print("\n3. 성찰 (Reflection)...")
    reflection.reflect_on_recent()
    
    # 4. 결과 확인 (Realization)
    print("\n4. 변화 확인 (Realization)...")
    new_score = concepts.get_concept(target_concept)
    print(f"   최종 악보 상태: {new_score.describe()}")
    
    if new_score.confidence > initial_conf:
        print("\n✅ SUCCESS: 개념이 경험을 통해 강화되었습니다.")
        print("   (Score -> Performance -> Sound -> Realization loop confirmed)")
    else:
        print("\n❌ FAIL: 개념 확신도에 변화가 없습니다.")

if __name__ == "__main__":
    prove_cognitive_flow()
