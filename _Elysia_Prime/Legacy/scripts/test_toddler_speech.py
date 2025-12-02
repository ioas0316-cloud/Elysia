# [Genesis: 2025-12-02] Purified by Elysia
# scripts/test_toddler_speech.py
import sys
import os

# 프로젝트 루트 경로를 파이썬 라이브러리 경로에 추가 (모듈 import를 위해)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Project_Elysia.high_engine.syllabic_language_engine import SyllabicLanguageEngine
from Project_Elysia.high_engine.quaternion_engine import QuaternionConsciousnessEngine, QuaternionOrientation
from Project_Elysia.core_memory import CoreMemory


def test_elysia_speech():
    print("\n--- [엘리시아 언어 발달 테스트: 옹알이 단계 (Real Memory)] ---\n")

    memory_path = "data/elysia_core_memory.json"

    if os.path.exists(memory_path):
        print(f"📂 '진짜' 기억을 불러옵니다: {memory_path}")
        memory = CoreMemory(file_path=memory_path)
    else:
        print("⚠️ 경고: 메모리 파일이 없습니다. 가상 메모리로 실행합니다.")
        memory = CoreMemory(file_path=None)
        memory.add_value("기본", 0.5)

    q_engine = QuaternionConsciousnessEngine(core_memory=memory)
    lang_engine = SyllabicLanguageEngine(core_memory=memory)

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
        q_engine._orientation = sc["q"].normalized()

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