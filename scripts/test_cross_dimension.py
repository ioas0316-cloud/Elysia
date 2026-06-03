import sys
import os
import secrets
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.perception_engine import UniversalBinaryMapper
from core.knowledge_space import PhaseSpace

def run():
    print("🌌 Initializing Elysia's Auto-Boundary & Cross-Dimensional Phase Space...")
    space = PhaseSpace()

    # 1. 이종 데이터의 유입 (아무런 기준도, 라벨도 주어지지 않음)
    txt_kor = "이것은 사과라는 한글 텍스트 데이터입니다.".encode('utf-8')
    txt_eng = "This is a text data of an apple.".encode('utf-8')
    img_apple = secrets.token_bytes(60) # 무질서도가 높은 사과 이미지의 바이트 흉내
    img_banana = secrets.token_bytes(60) # 무질서도가 높은 바나나 이미지의 바이트 흉내

    print("\n[Phase 1] Ingesting Raw Binary Streams")
    space.add_concept(UniversalBinaryMapper.map(txt_kor, "Raw_Apple_1"))
    space.add_concept(UniversalBinaryMapper.map(txt_eng, "Raw_Apple_2"))
    space.add_concept(UniversalBinaryMapper.map(img_apple, "Raw_Apple_3"))
    space.add_concept(UniversalBinaryMapper.map(img_banana, "Raw_Banana_4"))

    # 2. 시스템 스스로 밀도(Density)를 측정하여 경계면(Event Horizon) 획정
    print("\n[Phase 2] System Automatically Discovering Boundaries based on Density")
    logs = space.discover_boundaries("Natural_Entropy")
    for log in logs:
        print("  =>", log)

    # 3. 인간의 라벨이 유입되어 특정 물리 군집을 교차차원화(Cross-Dimensionalize)
    print("\n[Phase 3] Human Labels Intersecting the Physical Boundaries")
    print("User says: 'Raw_Apple_1 is Text.'")
    res1 = space.cross_dimensionalize("Raw_Apple_1", "Text")
    print("  =>", res1)

    print("User says: 'Raw_Apple_3 is Image.'")
    res2 = space.cross_dimensionalize("Raw_Apple_3", "Image")
    print("  =>", res2)

    space.print_state()
    print("🏁 Simulation Complete. Human labels naturally mapped onto self-discovered physical boundaries.")

if __name__ == "__main__":
    run()
