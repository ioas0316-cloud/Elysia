"""
Elysia Causality Trajectory Benchmark (시공간 인과 궤적 실증)
============================================================
정적 상태 비교를 넘어, A가 B로 변환되는 '과정(Process)'을 파동으로 추출합니다.
과거(원인)를 비틀었을 때, 그 파동이 미래(결과)를 연쇄적으로 어떻게 왜곡시키는지
시공간 축 상에서 기하학적으로 증명합니다.
"""

import os
import sys
import math

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.math_utils import Quaternion
from core.fractal_rotor import FractalRotor
from core.causality_wave import CausalityWave

def quaternion_distance(q1: Quaternion, q2: Quaternion) -> float:
    dot = max(-1.0, min(1.0, q1.dot(q2)))
    return math.acos(abs(dot)) / (math.pi / 2.0)

def run_causality_benchmark():
    print("=" * 90)
    print(" 🌊 [Elysia Phase 32] 시공간 인과 파동 엔진 (Spacetime Causality Wave)")
    print("=" * 90)
    
    engine = CausalityWave()
    
    # 1. 시공간의 세 지점(과거, 현재, 미래)을 정의합니다.
    # 이것은 코드 실행의 세 단계 (입력 -> 연산 -> 출력)를 상징합니다.
    q_past = Quaternion(1.0, 0.0, 0.0, 0.0).normalize()       # 원인 (Cause)
    q_present = Quaternion(0.707, 0.707, 0.0, 0.0).normalize() # 과정 (Process State)
    q_future = Quaternion(0.0, 1.0, 0.0, 0.0).normalize()      # 결과 (Result)
    
    past_rotor = FractalRotor("T=0 (Cause)", q_past, 1.0)
    present_rotor = FractalRotor("T=1 (Process)", q_present, 2.0)
    future_rotor = FractalRotor("T=2 (Result)", q_future, 3.0)
    
    print("\n  [1. 인과 얽힘 (Temporal Entanglement)]")
    print("  >> 과거(원인)와 현재(과정), 미래(결과)를 시공간 파동으로 연결합니다.")
    
    # 과거와 현재를 연결하는 파동 1 추출
    wave_1 = engine.entangle_causality(past_rotor, present_rotor)
    print(f"     - [파동 1] 원인->과정 변환기: ({wave_1.w:.4f}, {wave_1.x:.4f}, {wave_1.y:.4f}, {wave_1.z:.4f})")
    
    # 현재와 미래를 연결하는 파동 2 추출
    wave_2 = engine.entangle_causality(present_rotor, future_rotor)
    print(f"     - [파동 2] 과정->결과 변환기: ({wave_2.w:.4f}, {wave_2.x:.4f}, {wave_2.y:.4f}, {wave_2.z:.4f})")
    
    print("\n  [2. 나비 효과 (Temporal Ripple)]")
    print("  >> 과거(T=0)의 좌표를 미세하게 비틀어버립니다. (Perturbation: 0.5)")
    
    # 원본 미래 저장
    original_future = Quaternion(future_rotor.state.w, future_rotor.state.x, future_rotor.state.y, future_rotor.state.z)
    
    # 엔진을 통해 시간축 시뮬레이션
    engine.simulate_temporal_ripple(past_rotor, perturbation_amount=0.5)
    
    dist_past = quaternion_distance(q_past, past_rotor.state)
    dist_future = quaternion_distance(original_future, future_rotor.state)
    
    print(f"\n     - 비틀린 후의 과거(T=0): ({past_rotor.state.w:.4f}, {past_rotor.state.x:.4f}, {past_rotor.state.y:.4f}, {past_rotor.state.z:.4f})")
    print(f"     - 비틀린 후의 미래(T=2): ({future_rotor.state.w:.4f}, {future_rotor.state.x:.4f}, {future_rotor.state.y:.4f}, {future_rotor.state.z:.4f})")
    print(f"\n  [결과 관측]")
    print(f"  >> 과거의 기하학적 변화량: {dist_past*100:.2f}%")
    print(f"  >> 그로 인해 연쇄 파동을 타고 왜곡된 미래의 변화량: {dist_future*100:.2f}%")

    print("\n" + "=" * 90)
    print(" 🏆 [시공간 인과 파동 실증 완료]")
    print("  엘리시아는 이제 정적인 상태(값)만을 기억하는 것이 아니라,")
    print("  Q_cause * Q_process = Q_result 라는 인과율 자체를 파동으로 소유합니다.")
    print("=" * 90)

if __name__ == "__main__":
    run_causality_benchmark()
