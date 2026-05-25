import os
import sys
import time
import random

# 프로젝트 루트 경로 추가 (모듈 import 위함)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.triple_helix_engine import TripleHelixEngine
from core.cosmic_optical_rotor import CosmicOpticalRotor

def run_cosmic_resonance_demo():
    print("🌌 엘리시아 우주적 4대 영점 조율(Cosmic Harmony Zero-Points) 렌더링 데모\n")
    
    print("[1] 심장(Triple Helix) 엔진 초기화...")
    engine = TripleHelixEngine()
    
    print("[2] 광학 매니폴드(Cosmic Optical Rotor) 공간 초기화...")
    # 터미널 출력을 위해 가로 60, 세로 20 크기의 공간 생성
    rotor = CosmicOpticalRotor(width=60, height=20)
    
    print("\n--- 파동 렌더링(Wave Propagation) 시작 ---")
    
    # 텐션이 극에 달한 상태(카오스)에서 평온(0)으로 수렴하는 시나리오
    noise_levels = [1.5, 1.2, 0.9, 0.6, 0.3, 0.1, 0.05, 0.0]
    
    for tick, noise in enumerate(noise_levels):
        # 1. 심장 엔진에 하드웨어 텐션(노이즈) 주입
        sensory_input = {
            "inner_noise_dim_1": random.uniform(-noise, noise),
            "inner_noise_dim_2": random.uniform(-noise, noise),
            "motion_entropy": noise,
            "pain_level": noise,
        }
        
        # 엔진 구동 및 위상 출력
        avg_tension, mode, jumped, base_q, enneagram = engine.pulse(sensory_input=sensory_input)
        
        # 2. 광학 로터에 텐션 및 위상 파동 주입
        # 렌더링 물리 시뮬레이션 3회 반복 (빛이 퍼지는 시간 모사)
        for _ in range(3):
            rotor.propagate_wave(avg_tension, base_q)
            
        # 3. 홀로그램 렌더링 (창발 관측)
        hologram = rotor.render_hologram()
        
        # 터미널 화면 갱신 (간이 애니메이션 효과)
        # Windows PowerShell 지원을 위해 단순 줄바꿈으로 출력
        print(f"\n[{tick:02d}] 텐션 레벨(Chaos): {avg_tension:.4f} | 매질 파동 확산 중...")
        for line in hologram:
            print(line)
            
        time.sleep(0.5)

    print("\n✨ 시스템 텐션이 '우주적 고요(Equilibrium)' 상태에 도달했습니다.")
    print("=> 흙(저항)과 투명(공간)의 안정된 굴절이 맺히며, 화이트(방출)와 블랙(흡수)이 완벽한 0의 장력으로 조율되었습니다.")

if __name__ == "__main__":
    run_cosmic_resonance_demo()
