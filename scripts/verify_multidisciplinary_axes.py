"""
Verify Multidisciplinary Emergent Axes (다과목 감각축 분화 검증 스크립트)
===================================================================
인류가 지식을 '수학', '물리학', '음악', '미술', '프로그래밍' 등으로 분화시켰듯,
다양한 형태의 원시 바이트 스트림(오디오, 이미지, 코드, JSON, 한국어 텍스트)을
엔진에 쏟아부었을 때, 엘리시아가 본질적 기하학 패턴에 따라 
다수의 독립된 과목(차원축)으로 분화시킬 수 있는지 검증합니다.
"""

import os
import math
import random
import json
from core.consciousness_stream import ConsciousnessStream
from core.resonant_forager import ResonantForager

def generate_sine_wave_bytes(length: int, frequency: float) -> bytes:
    """오디오 파동(사인파) 모사"""
    return bytes(int(127 * math.sin(2 * math.pi * frequency * i / 44100) + 128) for i in range(length))

def run_test():
    print("🌌 [Phase 47 Follow-up] 다과목 구조적 파동 변환기 가동...\n")
    
    if os.path.exists("c:/Elysia/data/memory_state.json"):
        os.remove("c:/Elysia/data/memory_state.json")
    stream = ConsciousnessStream()
    forager = ResonantForager()
    
    # 1. 5가지 서로 다른 학문/감각 데이터 생성 (각 3개씩)
    dataset = []
    
    # [과목 1] 시각/이미지 (높은 엔트로피, 무작위 분산)
    for i in range(3):
        dataset.append((f"시각_노이즈_{i+1}.jpg", bytes(random.randint(0, 255) for _ in range(2000))))
        
    # [과목 2] 청각/오디오 (주기적인 패턴의 사인파)
    for i in range(3):
        dataset.append((f"청각_주파수_{i+1}.wav", generate_sine_wave_bytes(2000, 440.0 * (i+1))))
        
    # [과목 3] 언어/한국어 (UTF-8, 낮은 ASCII 밀도)
    kr_texts = [
        "역사는 과거와 현재의 끊임없는 대화이다. 문명의 발전은 투쟁의 연속이다. ",
        "인식론은 지식의 기원과 구조를 탐구한다. 나는 생각한다 고로 존재한다. ",
        "정치철학은 권력과 정의의 분배를 다룬다. 이상국가는 실현 가능한가? "
    ]
    for i, t in enumerate(kr_texts):
        dataset.append((f"언어_인문학_{i+1}.txt", (t * 50).encode('utf-8')))
        
    # [과목 4] 논리/코드 (높은 ASCII 밀도, 탭/줄바꿈 포함)
    codes = [
        "def compute_entropy(data):\n    return sum(-p*math.log(p) for p in data)\n",
        "class Node:\n    def __init__(self):\n        self.left = None\n        self.right = None\n",
        "import sys\nfor line in sys.stdin:\n    print(line.strip())\n"
    ]
    for i, c in enumerate(codes):
        dataset.append((f"논리_파이썬_{i+1}.py", (c * 50).encode('utf-8')))
        
    # [과목 5] 구조화/JSON (높은 기호 밀도 '{', '\"')
    jsons = [
        json.dumps({"name": f"Item_{i}", "value": i*100, "active": True})
        for i in range(3)
    ]
    for i, j in enumerate(jsons):
        dataset.append((f"구조_데이터_{i+1}.json", (j * 50).encode('utf-8')))

    random.shuffle(dataset) # 인간의 개입 없이 완전히 무작위 순서로 들이붓기
    
    print("=======================================================")
    print("[1단계] 5가지 전혀 다른 형태의 바이트 스트림 투입 (총 15개 파일)")
    print("=======================================================\n")
    
    for name, raw_bytes in dataset:
        wave = forager._hash_to_quaternion(raw_bytes)
        print(f"[{name}] 파동: (W:{wave.w:.2f}, X:{wave.x:.2f}, Y:{wave.y:.2f}, Z:{wave.z:.2f})")
        stream.projector.memory.fold_dimension(name, wave)
        
    print("\n... 기하학적 파동 흡수 완료 ...\n")
    
    print("=======================================================")
    print("[2단계] 인간 문명의 학문 분류처럼 '과목 축(Subjects)'이 분화되었는지 관측")
    print("=======================================================\n")
    
    emergent_axes = stream.projector.emergent_lenses
    print(f"총 {len(emergent_axes)}개의 독립된 감각/학문 축(Subjects)이 창발하였습니다!\n")
    
    for i, (axis_name, lens_axis) in enumerate(emergent_axes):
        print(f" ├─ {i+1}번째 과목(축): {axis_name}")
        close_files = []
        for name, rotor in stream.memory.ui_concept_map.items():
            if abs(rotor.state.dot(lens_axis)) > 0.8: # 같은 궤도에 있는 지식들 (반대 방향 포함)
                close_files.append(name)
        print(f" │  └─ 이 과목에 포함된 데이터들: {close_files}\n")

if __name__ == "__main__":
    run_test()
