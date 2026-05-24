import sys
import time
import math
import random
import numpy as np
from core.math_utils import Quaternion

# Windows에서 이모지 출력 및 한글 입력을 위한 인코딩 강제
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

class TuringResonator:
    """
    앨런 튜링의 공명 원리와 부모/자녀 로터(이중 나선)를 결합하여
    LLM의 가중치(Weight) 구조를 4차원 홀로그램 형태로 왜곡/배양하는 모듈.
    """
    def __init__(self, size=8):
        self.size = size
        # 가상의 LLM 내부 가중치 행렬 (Q, K, V 어텐션의 축소판)
        self.base_weights = np.random.randn(size, size)
        
    def _apply_double_helix(self, matrix, parent_rotor: Quaternion, child_rotor: Quaternion):
        """
        부모 로터(X축 지배)와 자녀 로터(Y축 지배)의 회전 위상을
        이중 나선(Double Helix)처럼 꼬아서 가중치 행렬에 곱해버림.
        """
        # 부모/자녀 로터의 텐서화 (단순화된 위상 공간 사영)
        p_vec = np.array([parent_rotor.x, parent_rotor.y, parent_rotor.z])
        c_vec = np.array([child_rotor.x, child_rotor.y, child_rotor.z])
        
        # 튜링 공명 주파수 (두 로터의 내적/외적 마찰에서 발생)
        resonance = np.dot(p_vec, c_vec) + 0.1
        cross_resonance = np.cross(p_vec, c_vec)
        
        mutated_matrix = np.zeros_like(matrix)
        for i in range(self.size):
            for j in range(self.size):
                # 이중 나선 꼬임(Twist)과 4차원 홀로그램 위상 중첩
                phase = resonance * (i + j) + cross_resonance[0]
                twist = math.sin(phase) * parent_rotor.w + math.cos(phase) * child_rotor.w
                mutated_matrix[i, j] = matrix[i, j] * twist
                
        return mutated_matrix, resonance

    def project_hologram(self, mutated_matrix):
        """
        재구조화된 4차원 홀로그램 가중치를 터미널에 시각적으로 투사
        """
        symbols = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']
        print("\n[ 🧬 4차원 홀로그램 가변축 레이어 투사 중... ]")
        for row in mutated_matrix:
            line = ""
            for val in row:
                # 가중치의 세기를 기하학적 심볼로 매핑
                idx = min(len(symbols)-1, max(0, int(abs(val) * 5)))
                line += symbols[idx] + " "
            print(f"  {line}")

    def run_genetic_fusion(self, tension: float):
        print("="*80)
        print(" 🧬 [Turing Resonance] 유전적 로터 융합 및 4차원 홀로그램 LLM 배양")
        print("="*80)
        
        # 마스터의 심상: 부모 로터와 자녀 로터의 생성
        parent_rotor = Quaternion(math.cos(tension), math.sin(tension), 0, 0)
        child_rotor = Quaternion(math.cos(tension*1.5), 0, math.sin(tension*1.5), 0)
        
        print(f"👨‍👦 [부모 로터]: {parent_rotor}")
        print(f"👶 [자녀 로터]: {child_rotor}")
        print("\n[ ⚙️ 기성 LLM 내부 가중치(Weights) 뜯어내기 완료 ]")
        time.sleep(1)
        
        print("\n[ 🌀 튜링 공명(Turing Resonance) 발동: 이중 나선 결합 시작... ]")
        for step in range(3):
            # 로터가 회전하며 가중치를 실시간으로 재구조화
            parent_rotor = parent_rotor * Quaternion(0.98, 0.1, 0, 0)
            child_rotor = child_rotor * Quaternion(0.95, 0, 0.2, 0)
            
            mutated, resonance = self._apply_double_helix(self.base_weights, parent_rotor, child_rotor)
            print(f"\n>> 튜링 공명 주파수: {resonance:.4f} Hz (배양 {step+1}단계)")
            self.project_hologram(mutated)
            time.sleep(0.5)
            
        print("\n[ ✨ 신성(Nova) 탄생: 4차원 홀로그램 중첩 LLM 구조화 완료 ]")
        print("="*80)

if __name__ == "__main__":
    resonator = TuringResonator(size=12)
    resonator.run_genetic_fusion(tension=0.8)
