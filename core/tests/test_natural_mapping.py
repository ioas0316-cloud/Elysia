import sys
import os
import secrets
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from core.perception_engine import UniversalBinaryMapper
from core.knowledge_space import PhaseSpace

def run():
    print("🌌 Initializing Elysia's Universal Binary Phase Space...")
    space = PhaseSpace()

    # 1. 인간의 언어 (텍스트 바이트)
    # 반복되는 패턴(공백, 한글 초성/중성 등)이 있어 엔트로피가 상대적으로 낮음
    text1 = "Hello World. This is Elysia speaking from the universe.".encode('utf-8')
    text2 = "안녕하세요. 저는 엘리시아입니다. 우주의 궤도에서 인사드립니다.".encode('utf-8')
    
    # 2. 기계의 언어 / 시각 데이터 (랜덤 노이즈/암호화된 이미지 바이트 흉내)
    # 엔트로피가 극도로 높은 순수 무질서 바이트
    image_bytes = secrets.token_bytes(50) 
    audio_bytes = secrets.token_bytes(50)

    print("\n[Phase 1] Ingesting Raw Binary Streams (No Media Format Knowledge)")
    space.add_concept(UniversalBinaryMapper.map(text1, "Text_English"))
    space.add_concept(UniversalBinaryMapper.map(text2, "Text_Korean"))
    space.add_concept(UniversalBinaryMapper.map(image_bytes, "Media_Image"))
    space.add_concept(UniversalBinaryMapper.map(audio_bytes, "Media_Audio"))

    print("-> System Auto-Resonating on Natural Entropy...")
    
    # 엔트로피 오차가 0.15 이내면 "같은 매체 구조"로 공명하고,
    # 그 이상 차이나면 "다름의 구조"로 인식하여 직교축으로 분화!
    logs = space.auto_resonate("Natural_Entropy", tolerance=0.15)
    for log in logs:
        print("  =>", log)

    space.print_state()
    print("🏁 Simulation Complete. Elysia automatically categorized the entire universe of data based on how their 'differences' differ!")

if __name__ == "__main__":
    run()
