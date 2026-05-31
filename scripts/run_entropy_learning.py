"""
Elysia Entropy and Growth Learning (생명과 엔트로피의 파동 학습)
================================================================
씨앗이 나무가 되고 사과가 열리는 '성장(Growth)'의 과정과, 
사과가 썩어가는 '부패(Entropy)'의 과정을 시공간 인과 엔진으로 학습합니다.
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
from core.omni_modal_sensor import OmniModalSensor

def run_entropy_learning():
    print("=" * 90)
    print(" 🌱 [Elysia Phase 32-B] 생명(성장)과 엔트로피(부패)의 파동 학습")
    print("=" * 90)
    
    sensor = OmniModalSensor()
    engine = CausalityWave()
    
    # 텍스트를 위상(Quaternion)으로 변환하는 헬퍼 함수
    def text_to_rotor(text: str, time_step: int) -> FractalRotor:
        # OmniModalSensor의 _convert_bytes_to_rotor를 활용하여 텍스트 바이트를 파동으로 변환
        q_state = sensor._convert_bytes_to_rotor(text.encode('utf-8'))
        return FractalRotor(f"T={time_step}: {text}", q_state, time_step)

    print("\n  [1. 생명의 궤적 주입 (Trajectory of Growth)]")
    print("  >> '흙과 씨앗' -> '자라나는 나무' -> '열매 맺은 사과'의 순차적 흐름을 입력합니다.")
    
    # 시간에 따른 상태 로터 생성
    r_seed = text_to_rotor("흙 속의 씨앗 (Soil and Seed)", 0)
    r_tree = text_to_rotor("가지를 뻗은 나무 (Growing Tree)", 1)
    r_apple = text_to_rotor("탐스러운 사과 (Red Apple)", 2)
    r_rot = text_to_rotor("썩어가는 사과 (Rotting Apple, Entropy)", 3)
    
    # 인과 파동(과정) 추출 및 얽힘
    wave_growth_1 = engine.entangle_causality(r_seed, r_tree)
    wave_growth_2 = engine.entangle_causality(r_tree, r_apple)
    wave_decay = engine.entangle_causality(r_apple, r_rot)
    
    print(f"\n  [학습된 파동 (Learned Waves)]")
    print(f"  - 발아의 파동 (Seed->Tree) : ({wave_growth_1.w:.4f}, {wave_growth_1.x:.4f}, {wave_growth_1.y:.4f}, {wave_growth_1.z:.4f})")
    print(f"  - 결실의 파동 (Tree->Apple): ({wave_growth_2.w:.4f}, {wave_growth_2.x:.4f}, {wave_growth_2.y:.4f}, {wave_growth_2.z:.4f})")
    print(f"  - 부패의 파동 (Apple->Rot) : ({wave_decay.w:.4f}, {wave_decay.x:.4f}, {wave_decay.y:.4f}, {wave_decay.z:.4f})")
    
    print("\n  [2. 파동의 범용성 실증 (Wave Generalization)]")
    print("  >> 엘리시아는 이제 '부패(Decay)'라는 파동(과정) 자체를 소유했습니다.")
    print("  >> 만약 '싱싱한 포도'라는 새로운 상태에 이 [부패의 파동]을 곱하면 어떻게 될까요?")
    
    r_grape = text_to_rotor("싱싱한 포도 (Fresh Grape)", 0)
    
    # 포도에 부패의 파동을 적용 (결과 상태 예측)
    predicted_rot_grape_state = r_grape.state * wave_decay
    predicted_rot_grape_state = predicted_rot_grape_state.normalize()
    
    # 실제 '썩은 포도' 텍스트의 파동과 비교
    r_actual_rot_grape = text_to_rotor("썩어가는 포도 (Rotting Grape)", 1)
    
    # 위상 거리 계산
    dot = max(-1.0, min(1.0, predicted_rot_grape_state.dot(r_actual_rot_grape.state)))
    phase_distance = math.acos(abs(dot)) / (math.pi / 2.0)
    
    print(f"\n  - '싱싱한 포도' * [부패의 파동] = 예측된 썩은 포도 궤적")
    print(f"  - 실제 '썩어가는 포도' 텍스트의 고유 궤적과의 위상 차이(Distance): {phase_distance*100:.2f}%")
    print("  (이 거리가 가까울수록, 엘리시아가 엔트로피(부패)의 보편적 법칙을 기하학적으로 이해했음을 의미합니다.)")

    print("\n" + "=" * 90)
    print(" 🏆 [생명과 엔트로피 파동 학습 완료]")
    print("=" * 90)

if __name__ == "__main__":
    run_entropy_learning()
