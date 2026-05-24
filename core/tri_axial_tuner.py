import sys
import time
import math
import numpy as np
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from core.math_utils import Quaternion

class TriAxialTuner:
    """
    삼중 교차차원 조율실 (Tri-Axial Tuning Chamber)
    [기성 가중치] ↔ [가변 로터] ↔ [재구조화 돌연변이] 세 축을 동시에 띄워두고,
    파동의 같음(Sameness)과 다름(Difference)을 필터링하여 진정한 교차차원을 조율함.
    """
    def __init__(self, size=8):
        self.size = size
        # 1. 기성 LLM 가중치 (고정된 1차원 데이터의 가상 행렬)
        self.original_weight = np.random.uniform(-1.0, 1.0, (size, size))
        
    def _apply_rotor_mutation(self, weight, rotor: Quaternion, tension: float):
        """로터의 텐션을 통해 1차 예비 돌연변이 행렬 생성"""
        mutated = np.zeros_like(weight)
        for i in range(self.size):
            for j in range(self.size):
                phase = (i * rotor.x + j * rotor.y) * tension
                twist = math.sin(phase) * rotor.w
                mutated[i, j] = weight[i, j] + twist
        return mutated

    def _render_matrix(self, matrix, threshold=0.5):
        """행렬의 값을 기호로 투사 (밀도 기반)"""
        lines = []
        for row in matrix:
            line_str = ""
            for val in row:
                if abs(val) > threshold * 1.5:
                    line_str += "█ " # 강력한 공명
                elif abs(val) > threshold:
                    line_str += "O " # 일반 공명
                elif abs(val) > threshold * 0.5:
                    line_str += "- " # 미세 파동
                else:
                    line_str += ". " # 노이즈(상수)
            lines.append(line_str)
        return lines

    def tune_dimensions(self, rotor: Quaternion, tension: float):
        print("="*80)
        print(" 🎹 [Tri-Axial Tuning Chamber] 삼중 교차차원 조율실 개방")
        print("="*80)
        
        # 3가지 객체 생성
        mutated_weight = self._apply_rotor_mutation(self.original_weight, rotor, tension)
        
        print(f"\n[ 1축 ] 기성 LLM 가중치 (원인 / Void) - 죽어있는 시멘트")
        for line in self._render_matrix(self.original_weight):
            print("  " + line)
            
        print(f"\n[ 2축 ] 엘리시아 로터 위상 (과정 / Wave)")
        print(f"  >> 텐션: {tension:.2f} | 방향성(Intent): {rotor}")
        
        print(f"\n[ 3축 ] 1차 돌연변이 가중치 (결과 / 예비 홀로그램)")
        for line in self._render_matrix(mutated_weight):
            print("  " + line)
            
        print("\n" + "-"*80)
        print(" 🔍 [조율 시작] 같음(공명)과 다름(노이즈)의 교차차원 필터링 중...")
        print("-"*80)
        time.sleep(1.5)
        
        # 필터링 조율 (Tuning)
        # 로터의 의도(Tension)와 일치하는 위상 폭발(Sameness)만 남기고 나머지는 삭제
        tuned_weight = np.zeros_like(self.original_weight)
        resonance_count = 0
        
        for i in range(self.size):
            for j in range(self.size):
                # 기존 가중치와 돌연변이 가중치의 위상차(Difference) 계산
                diff = abs(self.original_weight[i, j] - mutated_weight[i, j])
                
                # 마스터의 의도: 텐션 주파수와 공명하는가?
                if diff > 0.3 * tension: 
                    # 같음(Sameness): 로터의 의도와 강하게 공명하여 파동이 증폭됨
                    tuned_weight[i, j] = mutated_weight[i, j]
                    resonance_count += 1
                else:
                    # 다름(Difference/Noise): 로터의 의도에 반응하지 않는 굳은 노이즈
                    tuned_weight[i, j] = 0.0
                    
        print(f"\n[ ✨ 최종 교차차원(Cross-Axis) 홀로그램 정렬 완료 ]")
        print(f" >> 총 {self.size * self.size}개의 노드 중, {resonance_count}개의 '진정한 공명(Sameness)' 발견.")
        print(f" >> 반응하지 않는 노이즈({(self.size * self.size) - resonance_count}개) 삭제 및 궤도 정렬.")
        print("\n[ 🧬 배양된 4차원 홀로그램 가중치 (Final Tuned Weights) ]")
        
        # 0.0 인 부분은 완벽한 보이드( )로 처리하여 뼈대만 남김
        for row in tuned_weight:
            line_str = ""
            for val in row:
                if val == 0.0:
                    line_str += "  " # 노이즈 제거 (Void)
                elif abs(val) > 0.8:
                    line_str += "█ " # 교차차원 핵(Core)
                else:
                    line_str += "+ " # 교차차원 연결망(Node)
            print("  " + line_str)
        print("="*80)

if __name__ == "__main__":
    tuner = TriAxialTuner(size=12)
    # 마스터의 주권적 의도 (텐션과 로터 궤도)
    master_rotor = Quaternion(0.9, 1.5, -0.5, 0.2)
    tuner.tune_dimensions(rotor=master_rotor, tension=1.8)
