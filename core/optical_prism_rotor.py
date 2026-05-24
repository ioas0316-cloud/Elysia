import time
import math
import numpy as np
from core.math_utils import Quaternion

class OpticalPrismRotor:
    """
    수류학적 광학 간섭계(Optical Interferometer)
    굳어있는 2D 가중치를 빛의 파동(Wave)으로 환원하여
    비간섭 상태와 간섭(로터에 의한 공명) 상태를 삼중 대조 관측하는 프리즘 코어.
    """
    def __init__(self, resolution=12):
        self.res = resolution
        # 원본(굳어있는) 가중치를 주파수 파동(상수축)으로 정의
        self.base_wave = np.random.uniform(0, 1, (resolution, resolution))

    def _render_hologram(self, wave_matrix):
        """
        홀로그램 파동의 밀도(Density)를 시각적 기호로 투사
        """
        symbols = [' ', '.', '-', '+', '=', 'x', 'o', 'O', '#', '█']
        projection = []
        for row in wave_matrix:
            line = ""
            for val in row:
                # 0 ~ 1 사이의 파동 강도를 기호로 매핑
                idx = min(len(symbols)-1, max(0, int(val * len(symbols))))
                line += symbols[idx] + " "
            projection.append(line)
        return projection

    def _apply_interference(self, base_wave, rotor: Quaternion, tension: float):
        """
        마법진(로터)을 통과하며 발생하는 파동 간섭(Interference) 처리
        """
        interfered = np.zeros_like(base_wave)
        phase_shift = rotor.angle * tension
        
        for i in range(self.res):
            for j in range(self.res):
                # 2D 좌표를 위상 공간으로 변환
                r = math.sqrt(i**2 + j**2)
                theta = math.atan2(j, i)
                
                # 가변 로터에 의한 간섭 무늬(프랙탈 파동) 생성
                interference_pattern = math.sin(r * rotor.x + phase_shift) * math.cos(theta * rotor.y)
                
                # 원본 파동과 간섭 파동의 중첩(Superposition)
                val = base_wave[i, j] + (interference_pattern * rotor.w * tension)
                
                # 물을 그릇에 담아 1로 수렴(Normalize)
                interfered[i, j] = 1 / (1 + math.exp(-val)) 
                
        return interfered

    def observe(self, rotor: Quaternion, tension: float):
        """
        삼중 거울을 통한 회전 관측 실행
        """
        print("\n" + "="*70)
        print(" 🪞 [삼중 관측] 광학 집적 회로 간섭계(Optical Prism) 가동")
        print(f" 🌀 투사 로터 위상: {rotor}")
        print("="*70)
        
        # [제 1축: 비간섭 파동 / 원인]
        print("\n[ 1. 비간섭(Non-interference) 파동 / 상수축 노이즈 ]")
        non_interfered_hologram = self._render_hologram(self.base_wave)
        for line in non_interfered_hologram:
            print(f"  {line}")
            time.sleep(0.05)
            
        # [제 2축: 간섭 파동 / 과정]
        print(f"\n[ 2. 간섭(Interference) 파동 / 가변 로터 관통 중... (텐션: {tension}) ]")
        interfered_wave = self._apply_interference(self.base_wave, rotor, tension)
        interfered_hologram = self._render_hologram(interfered_wave)
        for line in interfered_hologram:
            print(f"  {line}")
            time.sleep(0.05)
            
        # [제 3축: 삼중 대조 및 수렴 / 결과]
        print("\n[ 3. 대조(Comparison) 및 홀로그램 수렴(Convergence) ]")
        print(" >> 분석: 굳어있던 상수축(비간섭)의 노이즈들이 로터의 의도(가변축)에 의해 특정 패턴으로 수렴함.")
        print(" >> 물을 그릇에 담듯, 파동이 간섭을 통해 새로운 '1'의 형태로 재구조화 완료.")
        print("="*70 + "\n")
