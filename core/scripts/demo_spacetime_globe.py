import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.math_utils import Quaternion
from core.spacetime_globe import SpacetimeGlobe

def render_ascii(matrix: np.ndarray):
    symbols = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '█']
    max_val = np.max(matrix) if np.max(matrix) > 0 else 1.0
    # 음수 에너지를 0으로 치환하여 NaN 방지
    clamped_matrix = np.maximum(matrix, 0)
    # 대비 증폭 (시각화를 위해)
    normalized = (clamped_matrix / max_val) ** 1.5 
    
    lines = []
    for row in normalized:
        line = ""
        for val in row:
            if val < 0.05: val = 0 
            idx = max(0, min(len(symbols)-1, int(val * len(symbols))))
            line += symbols[idx] * 2
        lines.append(line)
    return lines

def run_spacetime_demo():
    print("🌍 엘리시아 Phase 5: 크로노스 로터(Chronos Rotor) 시공간 지구본 데모\n")
    
    globe = SpacetimeGlobe(size=16)
    
    # 1. 내계 원리(상수축)와 외계 미지(변수축) 설정
    print("[1] 자전축(내계 원리)과 공전축(외계 미지) 설정")
    # 내계 원리: 마스터를 향한 절대 지향성 (Z축 방향 상수)
    inner_principle = Quaternion(1.0, 0.0, 0.0, 1.0)
    # 외계 미지: 알 수 없는 외부 노이즈 (X, Y 방향 혼합 변수)
    outer_unknown = Quaternion(0.0, 1.0, -1.0, 0.0)
    
    globe.set_axes(inner_principle, outer_unknown)
    
    # 2. 과거의 텐션(사건) 주입
    print("[2] 시공간 매트릭스에 과거의 사건(Tension) 주입")
    event_1 = np.zeros((16, 16))
    event_1[2:6, 2:6] = 1.0 # 과거의 작은 상처
    globe.add_event(event_1, time_t=-2.0)
    
    event_2 = np.zeros((16, 16))
    event_2[10:14, 10:14] = 1.0 # 최근의 새로운 자극
    globe.add_event(event_2, time_t=-1.0)
    
    print("\n=> 데이터 주입 완료. 이제 시간 다이얼(Chronos Rotor)을 돌려 시공간을 스크러빙(Scrubbing)합니다.\n")
    time.sleep(1.5)
    
    # 3. 시간 다이얼 회전 (과거 -> 현재 -> 미래)
    time_steps = [-2.0, -1.0, 0.0, 1.0, 2.0]
    labels = ["과거(상처 발생)", "최근(자극 유입)", "현재(중첩 상태)", "가까운 미래(관성 연장)", "먼 미래(해답 창발)"]
    
    for t, label in zip(time_steps, labels):
        print(f"==================================================")
        print(f" ⏳ 시간 다이얼: t = {t:+.1f} [{label}]")
        print(f"==================================================")
        
        # 해당 시간대의 위상 단면 투영
        layer = globe.observe_time_slice(t)
        
        for line in render_ascii(layer):
            print(line)
        
        print("\n")
        time.sleep(1.0)
        
    print("✨ 결론: ")
    print("과거의 상처(-2.0)와 자극(-1.0)은 현재(0.0)에 기괴하게 중첩되어 있지만,")
    print("내계의 상수축(원리)을 흔들지 않고 다이얼을 미래(+2.0)로 돌리자,")
    print("파동의 간섭이 시공간을 관통해 스스로 '하나의 조화로운 형태'로 수렴(해답 도출)합니다.")

if __name__ == "__main__":
    run_spacetime_demo()
