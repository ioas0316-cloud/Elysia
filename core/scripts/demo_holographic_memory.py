import os
import sys
import numpy as np

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.math_utils import Quaternion
from core.holographic_manifold import HolographicMemoryMatrix

def render_ascii(matrix: np.ndarray, threshold=0.5):
    """2D 배열을 ASCII 기호로 렌더링"""
    symbols = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '█']
    max_val = np.max(matrix) if np.max(matrix) > 0 else 1.0
    normalized = matrix / max_val
    
    lines = []
    for row in normalized:
        line = ""
        for val in row:
            if val < 0: val = 0 # 음수 에너지는 0 처리
            idx = max(0, min(len(symbols)-1, int(val * len(symbols))))
            line += symbols[idx] * 2
        lines.append(line)
    return lines

def run_holographic_demo():
    print("🌌 엘리시아 Phase 4: 4D 홀로그래픽 위상 매트릭스 비충돌 데모\n")
    
    # 1. 16x16 크기의 메모리 매트릭스 생성
    memory = HolographicMemoryMatrix(size=16)
    
    # 2. 저장할 두 개의 상이한 데이터(이미지) 생성
    data_A = np.zeros((16, 16))
    data_B = np.zeros((16, 16))
    
    # Data A: 사각형 (Square)
    data_A[4:12, 4:12] = 1.0
    
    # Data B: 십자가 (Cross)
    data_B[2:14, 7:9] = 1.0
    data_B[7:9, 2:14] = 1.0
    
    print("[1] 두 개의 서로 다른 차원 축(Rotor) 정의")
    # 서로 직교하는 위상 주파수를 가져야 간섭이 캔슬됨
    rotor_A = Quaternion(1.0, 1.0, 0.0, 0.0).normalize() # X축 기반 파동
    rotor_B = Quaternion(1.0, 0.0, 1.0, 0.0).normalize() # Y축 기반 파동
    
    print("[2] 메모리 공간에 파동으로 치환하여 중첩(Superposition) 저장")
    memory.add_memory(data_A, rotor_A)
    memory.add_memory(data_B, rotor_B)
    
    print("\n=> 데이터 A(사각형)와 B(십자가)가 동일한 3D 복소수 공간에 중첩되었습니다.")
    # 내부 공간의 중심 단면(z=8) 실수부 출력 (알아볼 수 없는 노이즈 상태 확인)
    print("=> 렌즈 없이 본 메모리의 날것(Raw) 상태 (알아볼 수 없는 간섭무늬):")
    raw_slice = memory.matrix[8, :, :].real
    for line in render_ascii(raw_slice):
        print(line)
        
    print("\n" + "="*50)
    print("🔍 [관측 1] 로터 A(X축 주파수)를 비추어 '2D 평면 레이어'로 투영")
    print("="*50)
    layer_A = memory.project_2d_layer(rotor_A)
    for line in render_ascii(layer_A):
        print(line)
        
    print("\n" + "="*50)
    print("🔍 [관측 2] 로터 B(Y축 주파수)를 비추어 '2D 평면 레이어'로 투영")
    print("="*50)
    layer_B = memory.project_2d_layer(rotor_B)
    for line in render_ascii(layer_B):
        print(line)
        
    print("\n" + "="*50)
    print("🌍 [관측 3] 로터 A를 비추어 '3D 구체(Sphere) 표면'으로 둥글게 말아 투영")
    print("="*50)
    sphere_A = memory.project_3d_sphere(rotor_A)
    for line in render_ascii(sphere_A):
        print(line)
        
    print("\n✨ 결론: 데이터들은 메모리 주소를 분할하지 않고 한 공간에 '충돌 없이' 공존합니다.")
    print("관측하는 축(Rotor)에 따라 평면으로 펴지거나 구체로 말리며, 온전히 자신의 형태를 창발시킵니다.")

if __name__ == "__main__":
    run_holographic_demo()
