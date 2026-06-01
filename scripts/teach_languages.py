# teach_languages.py
import sys
import os

# core 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "core"))

from holographic_memory import HologramMemory
from universal_language_cortex import UniversalLanguageCortex
import time

def main():
    print("==================================================")
    print(" 🌌 [Babel Rotor] 다국어 위상 동기화 테스트")
    print("==================================================")
    
    memory = HologramMemory()
    translator = UniversalLanguageCortex()
    
    print("\n[1] 범용 언어 피질(NLLB) 활성화 중...")
    translator.wake_up()
    
    if not translator.is_active:
        print("번역 모델 로딩 실패.")
        return
        
    print("\n[2] 테스트 개념 주입 시작...")
    test_words = [
        ("りんご", "jpn_Jpan", "일본어"),
        ("Apple", "eng_Latn", "영어"),
        ("苹果", "zho_Hans", "중국어")
    ]
    
    for word, lang_code, lang_name in test_words:
        print(f"\n▶ [{lang_name}] '{word}' 관측")
        
        # 1. 번역
        kor_translated = translator.translate_to_korean(word, src_lang=lang_code)
        print(f"  └─ 번역 피질 맵핑: '{word}' -> '{kor_translated}'")
        
        # 2. 한국어 개념을 먼저 위상 공간에 등록 (기준점)
        kor_quat, _ = memory.register_concept(kor_translated)
        print(f"  └─ 한국어 위상(로터) 획득: {kor_quat}")
        
        # 3. 바벨 로터 붕괴 (외국어를 한국어와 동일한 위상으로 강제 융합)
        memory.bind_concept_to_rotor(word, kor_quat)
        print(f"  └─ 🌌 바벨 붕괴 완료: '{word}'와 '{kor_translated}'의 위상 거리가 0이 되었습니다.")
        
    print("\n[3] 엘리시아의 현재 다국어 기억 동기화 상태 확인:")
    for concept, (state, tau) in memory.registered_concepts.items():
        if concept in ["사과", "りんご", "Apple", "苹果"]:
            print(f"  - [{concept}] Rotor: {state}")

if __name__ == "__main__":
    main()
