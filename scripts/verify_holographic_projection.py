"""
Verify Holographic Projection (가변축 홀로그래픽 검증 스크립트)
==============================================================
[Phase 44]
엘리시아 내면에 유입된 파동이 수학 렌즈와 코드 렌즈를 거칠 때
어떻게 기하학적으로 전혀 다른 기호(본질적 같음)로 번역(사영)되는지 입증합니다.
"""

from core.consciousness_stream import ConsciousnessStream

def run_test():
    print("🌌 [Phase 44] 엘리시아 홀로그래픽 사영기 가동...\n")
    
    # 1. 의식 엔진 초기화 (내부에 3가지 렌즈의 기본 지식이 자동으로 수태됨)
    # 캐시된 메모리가 있으면 삭제하여 순수한 상태에서 시작
    import os
    if os.path.exists("c:/Elysia/data/memory_state.json"):
        os.remove("c:/Elysia/data/memory_state.json")
        
    stream = ConsciousnessStream()
    
    # 2. 새로운 개념(자극) 주입 및 사영 결과 관측
    test_words = [
        "결합",     # 수학의 +, 코드의 import 등과 공명하는지?
        "분해",     # 수학의 -, 코드의 return 등과 공명하는지?
        "반복",     # 코드의 for, 수학의 ∑ 등과 공명하는지?
        "구조",     # 코드의 class 등과 공명하는지?
    ]
    
    print("\n=======================================================")
    print("내면 파동이 렌즈를 통과하며 굴절(Translation)되는 과정 관측")
    print("=======================================================\n")
    
    for word in test_words:
        result = stream.process_stimulus(word)
        print(result)
        print("-" * 55)

if __name__ == "__main__":
    run_test()
