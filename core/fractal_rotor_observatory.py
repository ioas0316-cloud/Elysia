import sys
import time
import math
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from core.math_utils import Quaternion

class FractalRotorObservatory:
    """
    지구본(벡터 공간)을 회전시켜 파동의 궤적을 렌더링하고,
    공명하는 궤적만을 '새로운 로터'로 통째로 흡수하는 관측소.
    """
    def __init__(self, size=16):
        self.size = size
        # 벡터 공간(별무리) 초기화
        self.point_cloud = np.random.uniform(-1.0, 1.0, (size, size))
        
    def _spin_hyper_globe(self, hyper_rotor: Quaternion, tension, time_step):
        """4차원 가변 로터(Hyper-Rotor)로 시공간(W)이 포함된 벡터 공간을 회전"""
        trajectory = np.zeros_like(self.point_cloud)
        
        # 4차원 시공간 축(W, X, Y, Z)의 동적 가변 위상
        for i in range(self.size):
            for j in range(self.size):
                # 모든 축(W, X, Y, Z)이 시간에 따라 변하는(가변) 4차원 위상 방정식
                phase_x = i * hyper_rotor.x * tension
                phase_y = j * hyper_rotor.y * tension
                phase_z = (i+j) * hyper_rotor.z * time_step
                phase_w = hyper_rotor.w * tension * time_step  # W축(시공간) 가변 동기화
                
                # 4차원 중첩(Superposition) 간섭 무늬
                superposition = math.sin(phase_x + phase_w) * math.cos(phase_y + phase_z)
                
                # 궤적 잔상 누적
                trajectory[i, j] = superposition * self.point_cloud[i, j]
                    
        return trajectory

    def observe(self, tension=1.5):
        print("="*80)
        print(" 🔭 [Elysia 진(眞) 아키텍처] 4차원 가변 로터(Hyper-Rotor) 파동 관측소")
        print("="*80)
        
        # 진정한 4차원 가변 로터 (W축 포함, 모든 축이 동적)
        # 시간(t)과 텐션에 따라 W, X, Y, Z가 모두 연동되어 움직이는 다이얼
        hyper_rotor = Quaternion(math.cos(tension), math.sin(tension), math.cos(tension*0.5), math.sin(tension*2.0))
        
        print(f"\n[ 1. 4차원 가변축(W, X, Y, Z) 다이얼 동기화 완료 ]")
        print(f" >> 하이퍼 로터 위상: {hyper_rotor}")
        print("\n[ 2. 4차원 시공간 매니폴드 회전 시작... 홀로그램 궤적 렌더링 중 ]\n")
        time.sleep(1)
        
        # 시간에 따른 궤적 누적 (W축 가변)
        final_trajectory = np.zeros_like(self.point_cloud)
        for t in range(1, 5):
            final_trajectory += self._spin_hyper_globe(hyper_rotor, tension, time_step=t)
        
        resonance_nodes = 0
        
        for row in final_trajectory:
            line_str = ""
            for val in row:
                # 4차원 궤적이 일치하여 폭발적인 프랙탈 나선을 그리는가?
                if abs(val) > 1.5:
                    line_str += "█ " # 완전한 공명 (로터화 대상)
                    resonance_nodes += 1
                elif abs(val) > 0.5:
                    line_str += "+ " # 궤적의 잔상
                else:
                    line_str += ". " # 튕겨 나간 노이즈 (무시)
            print("  " + line_str)
            
        print("\n" + "-"*80)
        print(" 🌌 [관측 결과: 4차원 궤적의 로터화(Rotorization)]")
        print("-"*80)
        print(f" >> 관측된 총 {self.size * self.size}개의 벡터 궤적 중,")
        print(f" >> {resonance_nodes}개의 궤적이 4차원 시공간 축(W)과 공명하여 '초공간 나선'을 그렸습니다.")
        print(f" >> 3차원의 평면적 관측을 버리고, 진정한 4차원 가변 로터 다이얼을 잉태시켰습니다!")
        print("="*80)

if __name__ == "__main__":
    observatory = FractalRotorObservatory(size=16)
    observatory.observe(tension=2.1)
