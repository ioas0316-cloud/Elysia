"""
Verify Omni-Modal Emergent Axes (옴니모달 감각축 창발 검증 스크립트)
===================================================================
[Phase 47]
텍스트(언어)의 바이트 스트림과 이미지(시각)의 원시 바이트 스트림을 차별 없이 엔진에 투입했을 때,
하드코딩된 필터 없이도 '바이트 패턴의 본질적 다름'으로 인해 자연스럽게
'텍스트 축'과 '이미지 축'이 분화하여 창발하는지 증명합니다.
"""

import os
import random
from core.consciousness_stream import ConsciousnessStream
from core.resonant_forager import ResonantForager

def run_test():
    print("🌌 [Phase 47] 옴니모달 구조적 파동 변환기 가동...\n")
    
    # 1. 엔진 및 탐색기 초기화
    if os.path.exists("c:/Elysia/data/memory_state.json"):
        os.remove("c:/Elysia/data/memory_state.json")
    stream = ConsciousnessStream()
    forager = ResonantForager()
    
    print("=======================================================")
    print("[1단계] 텍스트(.txt) 및 이미지(.jpg) 원시 바이트 생성 및 투입")
    print("=======================================================\n")
    
    # 더미 텍스트 데이터 (엔트로피 낮음, 긴 한국어/영어 문장)
    texts = [
        ("문서_철학.txt", ("존재와 무, 그리고 인식에 대한 고찰. 철학은 세계를 이해하는 방식이다. " * 50).encode('utf-8')),
        ("문서_물리.txt", ("양자역학은 미시 세계의 확률적 상태를 기술한다. E=mc^2. " * 50).encode('utf-8')),
        ("문서_역사.txt", ("인류의 발자취와 문명의 흥망성쇠를 기록하다. 역사는 반복된다. " * 50).encode('utf-8')),
        ("문서_코드1.txt", ("def calculate(a, b):\n    return a + b\nprint('Hello World')\n" * 50).encode('utf-8')),
        ("문서_코드2.txt", ("import numpy as np\narr = np.array([1, 2, 3])\nprint(arr)\n" * 50).encode('utf-8')),
        ("문서_코드3.txt", ("class Rotor:\n    def __init__(self):\n        self.state = 0\n" * 50).encode('utf-8'))
    ]
    
    # 더미 이미지 데이터 (압축된 바이너리 모사: 완전한 랜덤 바이트 2000개)
    images = []
    for i in range(3):
        raw_bytes = bytes(random.randint(0, 255) for _ in range(2000))
        images.append((f"사진_풍경{i+1}.jpg", raw_bytes))
        
    all_data = texts + images
    random.shuffle(all_data) # 무작위 순서로 투입 (구분 없이)
    
    for name, raw_bytes in all_data:
        wave = forager._hash_to_quaternion(raw_bytes)
        print(f"[{name}] 파동: ({wave.w:.2f}, {wave.x:.2f}, {wave.y:.2f}, {wave.z:.2f})")
        stream.projector.memory.fold_dimension(name, wave)
        
    print("\n... 옴니모달 바이트 투입 완료 ...\n")
    
    print("=======================================================")
    print("[2단계] 자생적 감각축(Sensory Axes) 창발 스캔")
    print("=======================================================\n")
    
    emergent_axes = stream.projector.emergent_lenses
    
    print(f"총 {len(emergent_axes)}개의 독립된 감각축(군집)이 창발하였습니다!\n")
    
    # 각 축과 가장 위상이 비슷한 노드들을 출력
    for i, (axis_name, lens_axis) in enumerate(emergent_axes):
        print(f" ├─ {i+1}번째 감각축: {axis_name}")
        # 이 축 근처에 있는 개념(파일)들 찾기
        close_files = []
        for name, rotor in stream.memory.ui_concept_map.items():
            if abs(rotor.state.dot(lens_axis)) > 0.75: # 축과 강하게 일치하는 것들 (반대 방향 포함)
                close_files.append(name)
        print(f" │  └─ 이 감각축에 매달린 데이터들: {close_files}")

    print("\n[결론]")
    print("만약 텍스트(.txt)들과 이미지(.jpg)들이 섞이지 않고 서로 다른 축으로 분화되었다면,")
    print("엘리시아는 시각과 언어를 가르치지 않아도 스스로 '다름'을 인지하고 분류해 낸 것입니다.")

if __name__ == "__main__":
    run_test()
