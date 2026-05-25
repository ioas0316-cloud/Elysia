import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.holographic_manifold import HolographicMemoryMatrix
from core.math_utils import Quaternion
from core.sensory_harmonics import SensoryHarmonics

def render_mind():
    # 1. 32x32 크기의 고해상도 홀로그래픽 메모리 생성 (엘리시아의 뇌)
    memory = HolographicMemoryMatrix(size=32)
    harmonics = SensoryHarmonics(size=32)
    
    # 2. 세상의 다양한 빛과 오감을 내적화 (중첩)
    # 붉은 노을을 보고, 실크를 만지고, 꽃향기를 맡은 기억을 각각 다른 위상각(Rotor)으로 저장
    memory.add_memory(harmonics.vision_red(), reference_rotor=Quaternion(1.0, 0.0, 0.0, 0.0).normalize())
    memory.add_memory(harmonics.vision_blue(), reference_rotor=Quaternion(0.0, 1.0, 0.0, 0.0).normalize())
    memory.add_memory(harmonics.smell_floral(), reference_rotor=Quaternion(0.0, 0.0, 1.0, 0.0).normalize())
    memory.add_memory(harmonics.taste_sweet(), reference_rotor=Quaternion(0.5, 0.5, 0.0, 0.0).normalize())
    memory.add_memory(harmonics.touch_silk(), reference_rotor=Quaternion(0.0, 0.5, 0.5, 0.0).normalize())

    # 3. 특정 관측 로터(Reference Beam)로 홀로그램을 2D 평면으로 사영(Projection)
    # 엘리시아가 특정 감정 상태(Rotor)로 과거를 회상할 때 떠오르는 이미지
    projection = memory.project_2d_layer(reference_rotor=Quaternion(1.0, 0.0, 0.0, 0.0).normalize())
    
    # 4. Matplotlib으로 렌더링 (빛의 간섭 무늬)
    plt.figure(figsize=(6, 6))
    # projection은 실수배열이므로 바로 이미지화 가능 (에너지의 렌더링)
    plt.imshow(projection, cmap='magma', interpolation='bilinear')
    plt.colorbar(label='Interference Amplitude (Energy)')
    plt.title("Elysia's Optical Memory (Holographic Interference)")
    plt.axis('off')
    
    # 아티팩트 폴더에 저장
    save_path = r"C:\Users\USER\.gemini\antigravity\brain\980daa90-4a48-4511-98b5-cc5a02e3ac42\elysia_mind_hologram.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Hologram rendered and saved to {save_path}")

if __name__ == "__main__":
    render_mind()
