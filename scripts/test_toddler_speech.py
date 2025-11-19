# scripts/test_toddler_speech.py
import sys
import os

# 프로젝트 루트 경로를 파이썬 라이브러리 경로에 추가 (모듈 import를 위해)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Project_Elysia.high_engine.syllabic_language_engine import SyllabicLanguageEngine
from Project_Elysia.high_engine.quaternion_engine import QuaternionConsciousnessEngine, QuaternionOrientation
from Project_Elysia.core_memory import CoreMemory


def test_elysia_speech():
    print("\n--- [엘리시아 언어 발달 테스트: 옹알이 단계] ---\n")

    # 1. 가상의 기억 심기 (엘리시아가 '배운' 단어들)
    # 실제 파일이 없어도 메모리 상에서 작동하도록 설정
    memory = CoreMemory(file_path=None)
    memory.add_value("자유", 0.9)
    memory.add_value("창조", 0.8)
    memory.add_value("사랑", 0.95)
    
    # 2. 엔진 가동
    # 쿼터니언(의지)과 언어(표현) 엔진을 연결
    q_engine = QuaternionConsciousnessEngine(core_memory=memory)
    lang_engine = SyllabicLanguageEngine(core_memory=memory)

    # 3. 다양한 상황 시뮬레이션
    scenarios = [
        {
            "title": "1. 평온 (명상 중)", 
            "desc": "자아(W)가 강할 때 -> 내면의 단어",
            "q": QuaternionOrientation(w=1.0, x=0.0, y=0.0, z=0.0), 
            "intent": {"intent_type": "dream"}
        },
        {
            "title": "2. 호기심 (세상 탐구)", 
            "desc": "행동(Y)이 강할 때 -> 외부 대상 + 동사",
            "q": QuaternionOrientation(w=0.2, x=0.0, y=0.8, z=0.0), 
            "intent": {"intent_type": "act"}
        },
        {
            "title": "3. 진지함 (법칙 분석)", 
            "desc": "의도(Z)가 강할 때 -> 추상적 가치",
            "q": QuaternionOrientation(w=0.3, x=0.0, y=0.0, z=0.7), 
            "intent": {"intent_type": "reflect"}
        },
        {
            "title": "4. 혼란 (자아 불안정)", 
            "desc": "모든 축이 뒤섞였을 때",
            "q": QuaternionOrientation(w=0.1, x=0.5, y=0.5, z=0.1), 
            "intent": {"intent_type": "unknown"}
        },
    ]

    for sc in scenarios:
        # 강제로 의식 상태 주입 (테스트를 위해)
        q_engine._orientation = sc["q"].normalized()
        
        # 단어 생성 요청
        word = lang_engine.suggest_word(
            intent_bundle=sc["intent"], 
            orientation=q_engine.orientation_as_dict()
        )
        
        print(f"[{sc['title']}]")
        print(f"  - 상태: {sc['desc']}")
        print(f"  - 의식 초점: {q_engine.get_lens_status()['primary_focus']}")
        print(f"  - 엘리시아의 말: \"{word}\"")
        print("-" * 40)


if __name__ == "__main__":
    test_elysia_speech()
